import rasterio
from rasterio.enums import Resampling
from osgeo import gdal
from osgeo import osr
import numpy as np
import numpy.ma as ma
import pandas as pd
from scipy.ndimage import uniform_filter
from sklearn.ensemble import RandomForestRegressor
from scipy.interpolate import griddata
from pykrige.rk import Krige
#from pykrige.compat import GridSearchCV
from pykrige.ok import OrdinaryKriging
from sklearn.model_selection import cross_val_score, GridSearchCV
import geopandas as gpd
from shapely.geometry import mapping
from rasterio.mask import mask as MASK
import glob
import time
from netCDF4 import Dataset
from numpy import dtype
from datetime import date, timedelta
import random
from scipy import stats


# In[5]:


def gen_tiff(array, lat, lon, out_file):
    xmin, ymin, xmax, ymax = [lon.min(), lat.min(), lon.max(), lat.max()]
    nrows, ncols = np.shape(array)
    xres = (xmax - xmin) / (float(ncols) - 1)
    yres = (ymax - ymin) / (float(nrows) - 1)
    xmin = xmin - xres / 2
    xmax = xmax - xres / 2
    ymin = ymin + yres / 2
    ymax = ymax + yres / 2
    geotransform = (xmin, xres, 0, ymax, 0, -yres)
    # That's (top left x, w-e pixel resolution, rotation (0 if North is up),
    #         top left y, rotation (0 if North is up), n-s pixel resolution)
    # I don't know why rotation is in twice???

    output_raster = gdal.GetDriverByName('GTiff').Create(out_file, ncols, nrows, 1, gdal.GDT_Float32)  # Open the file
    output_raster.SetGeoTransform(geotransform)  # Specify its coordinates

    srs = osr.SpatialReference()  # Establish its coordinate encoding
    srs.ImportFromEPSG(4326)  # This one specifies WGS84 lat long.
    # Anyone know how to specify the
    # IAU2000:49900 Mars encoding?
    output_raster.SetProjection(srs.ExportToWkt())  # Exports the coordinate system
    # to the file
    output_raster.GetRasterBand(1).WriteArray(array)  # Writes my array to the raster
    output_raster.GetRasterBand(1).SetNoDataValue(0)
    output_raster.FlushCache()
def day_avg(x, w):
    np.mean(x.reshape(-1, 3), axis=1)

def rfr_model(X, y):
    random.seed(31)
    np.random.seed(31)
    gsc = GridSearchCV(
        estimator=RandomForestRegressor(random_state=0),
        param_grid={
            'max_depth': [2, 4, 6, 8, 10, 12],
            'n_estimators': [1, 5, 10],
        },
        cv=3, scoring='neg_mean_absolute_error', verbose=False, n_jobs=16)
    grid_result = gsc.fit(X, y)
    best_params = grid_result.best_params_

    rfr = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"],
                                random_state=0, verbose=False)
    #     scores = cross_val_score(rfr, X, y, cv=3, scoring='neg_mean_absolute_error')
    regr = rfr.fit(X, y)
    return regr

def per_day(era5_t2m_day, era5land_t2m_day, era5_lat, era5_lon, era5land_lat, era5land_lon, dem, dem_diff, aspect,
            urban, ocean_mask, day):
    era5_t2m = era5_t2m_day
    era5land_t2m = era5land_t2m_day

    upscale_factor = 2.5
    gen_tiff(era5_t2m, era5_lat, era5_lon, out_file)
    with rasterio.open(out_file) as dataset:
        # resample data to target shape
        era5_t2m_re = dataset.read(
            out_shape=(
                dataset.count,
                len(era5land_lon),
                len(era5land_lat)
            ),
            resampling=Resampling.cubic
        )
    dataset.close()
    era5_t2m_re = era5_t2m_re[0, :]

    # mask
    mask = np.ma.mask_or(era5land_t2m.mask, ocean_mask.mask)

    # dem
    dem = ma.masked_array(dem, mask)
    dem_diff = ma.masked_array(dem_diff, mask)
    aspect = ma.masked_array(aspect, mask)
    urban = ma.masked_array(urban, mask)

    # reshaping
    dem = dem.reshape(dem.size, 1)
    dem_diff = dem_diff.reshape(dem_diff.size, 1)
    aspect = aspect.reshape(aspect.size, 1)
    era5land = era5land_t2m.reshape(era5land_t2m.size, 1)
    era5re = era5_t2m_re.reshape(era5_t2m_re.size, 1)

    # mask reshaped results
    era5land = era5land[dem.mask == False]
    era5re = era5re[dem.mask == False]
    dem = dem[dem.mask == False]
    dem_diff = dem_diff[dem_diff.mask == False]
    aspect = aspect[aspect.mask == False]

    # regression
    X = np.vstack((era5re, dem, dem_diff, aspect)).T
    y = era5land
    # reg = LinearRegression().fit(X, y)
    reg = rfr_model(X, y)
    pred = reg.predict(X)

    # compute error
    error = pred - y
    rmse = np.sqrt(np.mean(np.square(error)))

    # compute error in urban areas
    urban = urban.reshape(urban.size, 1)
    urban = urban[urban.mask == False]

    error_urban = error[urban == 1]
    rmse_urban = np.sqrt(np.mean(np.square(error_urban)))

    return reg, rmse, rmse_urban, era5_t2m_re
def predict(reg, era5_t2m_re, dem, dem_diff, aspect, ocean_mask):
    #     dem = ma.masked_array(dem,era5_t2m_re.mask)
    #     dem_diff = ma.masked_array(dem_diff,era5_t2m_re.mask)
    shape = dem.shape
    dem = dem.reshape(dem.size, 1)
    dem_diff = dem_diff.reshape(dem_diff.size, 1)
    aspect = aspect.reshape(aspect.size, 1)
    era5_t2m_re = era5_t2m_re.reshape(era5_t2m_re.size, 1)
    X = np.concatenate((era5_t2m_re, dem, dem_diff, aspect), axis=1)
    return ma.masked_array(reg.predict(X).reshape(shape), ocean_mask.mask)

def krig_error(pred, obs, era5land_lon, era5land_lat, urban, ocean_mask):
    # krig on error
    error = obs - pred
    shape = error.shape

    xx, yy = np.meshgrid(era5land_lon, era5land_lat)
    xx = xx[error.mask == False].astype(float)
    yy = yy[error.mask == False].astype(float)

    error_krig = error[error.mask == False]

    # parameter tuning for krig
    param_dict = {
        "method": ["ordinary"],
        "variogram_model": ["linear", "gaussian"],
        "nlags": [10, 50],
        "n_closest_points": [50, 100]
        # "weight": [True, False]
    }
    random.seed(31)
    np.random.seed(31)
    estimator = GridSearchCV(Krige(), param_dict, verbose=False, n_jobs=16, cv=3, scoring='neg_mean_absolute_error')
    # run the gridsearch
    X = np.column_stack((xx, yy))
    estimator.fit(X=X, y=error_krig)
    #     if hasattr(estimator, "best_score_"):
    # #         print("best_score mean_absolute_error = {:.3f}".format(-estimator.best_score_))
    # #         print("best_params = ", estimator.best_params_)
    OK = OrdinaryKriging(xx, yy, error_krig,
                         variogram_model=estimator.best_params_.get('variogram_model'),
                         verbose=False, enable_plotting=False, coordinates_type='geographic'
                         , nlags=estimator.best_params_.get('nlags'))
    xx, yy = np.meshgrid(era5land_lon, era5land_lat)
    xx = xx[ocean_mask.mask == False].astype(float)
    yy = yy[ocean_mask.mask == False].astype(float)
    z, ss = OK.execute('points', xx, yy, n_closest_points=estimator.best_params_.get('n_closest_points'), backend='C')
    krig_base = np.zeros(era5land_t2m_day.shape)
    krig_base[ocean_mask.mask == False] = z

    pred2 = pred + krig_base
    error2 = obs - pred2
    rmse2 = np.sqrt(np.mean(np.square(error2)))
    error2_urban = ma.masked_array(error2, urban == 1)
    rmse2_urban = np.sqrt(np.mean(np.square(error2_urban)))
    return rmse2, rmse2_urban, error2, pred2, z, estimator.best_params_, OK
def write_nc(file_name, era5land_lat, era5land_lon, era5_day_out, imputed_out):
    # write to netcdf
    ncout = Dataset(file_name, 'w', format='NETCDF4')
    # define axis size
    ncout.createDimension('time', None)  # unlimited
    ncout.createDimension('lat', len(era5land_lat))
    ncout.createDimension('lon', len(era5land_lon))
    # create time axis
    time = ncout.createVariable('time', dtype('int').char, ('time',))
    time.long_name = 'time'
    time.units = 'days since 1900-01-01'
    time.calendar = 'gregorian'
    time.axis = 'T'

    # create latitude axis
    lat = ncout.createVariable('lat', dtype('float').char, ('lat'))
    lat.standard_name = 'latitude'
    lat.long_name = 'latitude'
    lat.units = 'degrees_north'
    lat.axis = 'Y'

    # create longitude axis
    lon = ncout.createVariable('lon', dtype('float').char, ('lon'))
    lon.standard_name = 'longitude'
    lon.long_name = 'longitude'
    lon.units = 'degrees_east'
    lon.axis = 'X'

    # create variable array
    vout = ncout.createVariable('t2m', dtype('double').char, ('time', 'lat', 'lon',), fill_value=-32767)
    vout.long_name = '2 metre temperature'
    vout.units = 'K'

    # copy axis from original dataset
    time[:] = era5_day_out[:]
    lon[:] = era5land_lon[:]
    lat[:] = era5land_lat[:]
    vout[:, :, :] = imputed_out[:, :, :]
    ncout.close()


# In[3]:


# determine where we would see mssing values
missing_list = []
shapefile = gpd.read_file(r"C:\Users\maria\Documents\work\Berkeley\boundaries\L1AD\L1AD_w_Panama\level1_gcs_modify3.shp") # city boundaries 
miss_pix = gdal.Open(r'C:\Users\maria\Documents\work\Berkeley\data\imputation\ERA5land_mask.tif').ReadAsArray()
miss_pix = ma.masked_array(miss_pix, miss_pix == 0)
missing_df = pd.DataFrame()
idx=0
with rasterio.open(r'C:\Users\maria\Documents\work\Berkeley\data\imputation\ERA5land_mask.tif') as src:
    for i in shapefile.index:
        geoms = shapefile.loc[i].geometry
        geoms = [mapping(geoms)]
        out_image, out_transform = MASK(src, geoms, nodata=-9999, all_touched=True, pad=True)
        if 1 in out_image[0]:
            missing_list.append(shapefile.loc[i, 'SALID1'])
            missing_df.loc[idx,'SALID1'] = shapefile.loc[i, 'SALID1']
            missing_df.loc[idx,'missing']= np.count_nonzero(out_image[0] == 1)
            missing_df.loc[idx,'not_missing'] = np.count_nonzero(out_image[0] == 0)
            idx=idx+1
src.close()
missing_df['%missing'] = missing_df['missing']/(missing_df['missing']+missing_df['not_missing'])*100
missing_df.to_csv(r'C:\Users\maria\Documents\work\Berkeley\data\imputation\missing_summary_L1AD.csv')

#era5_land_ncs = glob.glob(r'D:\Work\ERA5LAND2001_2015\download_2000_Q4.nc') # specify "download_2000_Q4.nc" from the GDrive : this is the raw ERA5Land that has missing pixels
era5_land_ncs = glob.glob(r'C:\Users\maria\Downloads\ERA5_land_2016_Q2.nc') # specify "download_2000_Q4.nc" from the GDrive : this is the raw ERA5Land that has missing pixels

for era5_land_nc in era5_land_ncs:
    print(era5_land_nc)
    year = int(era5_land_nc.split('_')[-2])
    Q = int(era5_land_nc.split('_')[-1][1])
    #era5_nc = r'F:\YangJu\ERA5\ERA5_%s.nc' % year # this is the dataset (ERA5) from which missing pixels will be inferred (ie it doesn't have missing pixels)
    era5_nc = r'C:\Users\maria\Documents\work\Berkeley\data\imputation\ERA\ERA5_%s.nc' % year # this is the dataset (ERA5) from which missing pixels will be inferred (ie it doesn't have missing pixels)


    elevation = r'C:\Users\maria\Documents\work\Berkeley\data\imputation\ERA5Land_UX_DEM.tif'
    aspect_file = r'C:\Users\maria\Documents\work\Berkeley\data\imputation\ERA5Land_UX_ASPECT.tif'
    urban_mask = r'D:\Work\processing\irene_ft_raster\AD\AD_overall_binary.tif'
    out_file = r'C:\Users\maria\Documents\work\Berkeley\data\imputation\imputed\era5.tif'

    era5 = Dataset(era5_nc, mode='r')
    era5land = Dataset(era5_land_nc, mode='r')
    dem = gdal.Open(elevation).ReadAsArray()
    aspect = gdal.Open(aspect_file).ReadAsArray()
    urban = gdal.Open(urban_mask).ReadAsArray()
    dem_diff = dem - uniform_filter(dem, size=9)

    error_df = pd.DataFrame(columns=['SALID1', 'day', 'params'])
    error_df['params'] = error_df['params'].astype('object')
    row = 0
    f_date = date(year, 3 * Q - 2, 1)
    try:
        l_date = date(year, 3 * Q, 31)
    except:
        l_date = date(year, 3 * Q, 30)
    days = (l_date - f_date).days + 1
    print('#_days', days)
    era5land_lat = era5land.variables['latitude'][:]
    era5land_lon = era5land.variables['longitude'][:]
    era5_lat = era5.variables['latitude'][:]
    era5_lon = era5.variables['longitude'][:]

    era5_day_out = []
    imputed_out = np.empty([1, len(era5land_lon), len(era5land_lat)])

    for day in range(0,days):
        start_date = f_date + timedelta(days=day)
        print(start_date)
        start_hour = (start_date - date(1900, 1, 1)).days * 24
        end_hour = (start_date + timedelta(days=1) - date(1900, 1, 1)).days * 24
        era5land_time = np.arange(start_hour, end_hour)

        # slice data to the same day...
        era5land_time_mask = np.isin(era5land['time'], era5land_time)
        era5land_t2m = era5land['t2m'][era5land_time_mask, :]

        era5_time = era5['time'][:]
        era5_time_mask = np.isin(era5_time, era5land_time)
        era5_t2m = era5['t2m'][era5_time_mask, :]
        era5_time = era5_time[era5_time_mask]

        # calcualte daily mean
        era5_t2m_day = era5_t2m.mean(axis=0)
        era5land_t2m_day = era5land_t2m.mean(axis=0)

        # calcualte the #days since the start
        era5_day = np.unique(np.floor(era5_time / 24))[0]
        era5_day_out.append(era5_day)

        # iterate for all missing units within a day
        master = np.zeros(era5land_t2m_day.shape)
        master_count = np.zeros(era5land_t2m_day.shape)
        for missing_unit in missing_list:
            start = time.time()
            print('Working on day %s in unit %s...' % (era5_day, missing_unit))
            city_mask = gdal.Open(r'D:\Work\processing\irene_ft_raster\AD\by_ID\%s_AD_ID.tif' % missing_unit).ReadAsArray()
            city_mask = ma.masked_array(city_mask, city_mask != missing_unit)
            ocean_mask = city_mask.copy()
            city_mask[city_mask.mask == False] = 1
            x, y = np.where(ocean_mask == missing_unit)
            d = 15  # bbox size of the analysis
            ocean_mask[np.min(x) - d:np.max(x) + d, np.min(y) - d:np.max(y) + d] = missing_unit

            # RF
            reg, rmse, rmse_urban, era5_t2m_re = per_day(era5_t2m_day, era5land_t2m_day, era5_lat, era5_lon,
                                                         era5land_lat, era5land_lon, dem, dem_diff, aspect, urban,
                                                         ocean_mask, day)

            pred = predict(reg, era5_t2m_re, dem, dem_diff, aspect, ocean_mask)

            error_df.loc[row, 'SALID1'] = missing_unit
            error_df.loc[row, 'day'] = era5_day
            error_df.loc[row, 'rmse_RF'] = rmse
            error_df.loc[row, 'rmse_urban_RF'] = rmse_urban
            true_missing = pred * city_mask * miss_pix
            error_df.loc[row, 'rmse_urban_RF+Krig'] = 999
            k = 0
            extroplate_flag = 0
            try:
                error2, error2_urban, error_array, pred_array, krig_res, params, OK = krig_error(pred, era5land_t2m_day,
                                                                                                 era5land_lon,
                                                                                                 era5land_lat, urban,
                                                                                                 ocean_mask)
                error_df.loc[row, 'rmse_RF+Krig'] = error2
                error_df.loc[row, 'rmse_urban_RF+Krig'] = error2_urban
                error_df.at[row, 'params'] = params
                print('I did Kriging')
                k = 1
            except Exception as e:
                print('I failed Krig', e)
                pass
            if (error_df.loc[row, 'rmse_urban_RF+Krig'] <= error_df.loc[row, 'rmse_urban_RF']) & (k == 1):
                print('Kriging in results')
                krig_base = np.zeros(era5land_t2m_day.shape)
                krig_base[ocean_mask.mask == False] = krig_res
                pred_krig = pred + krig_base
                true_missing_krig = pred_krig * city_mask * miss_pix
                # check extroplated values if problematic (abs>1)
                extroplated = krig_base[true_missing_krig.mask == False]
                extroplated_min = stats.describe(extroplated).minmax[0]
                extroplated_max = stats.describe(extroplated).minmax[1]
                if extroplated_min < -1 or extroplated_max > 1:
                    final = pred * city_mask * miss_pix
                    extroplate_flag = 1
                    print('extroplation warning')
                else:
                    final = true_missing_krig
                del krig_res
            elif (error_df.loc[row, 'rmse_urban_RF+Krig'] > error_df.loc[row, 'rmse_urban_RF']) | (k == 0):
                print('No Kring in results')
                final = true_missing
            end = time.time()
            error_df.loc[row, 'time_spend'] = end - start
            error_df.loc[row, 'Krig_extroplate_flag'] = extroplate_flag
            error_df.to_csv(r'C:\Users\maria\Documents\work\Berkeley\data\imputation\ERROR_V2\ERROR_%s_%s.csv' % (year, Q))
            row = row + 1
            master = master + final.filled(0)
            final_count = final.mask == False
            master_count = master_count + final_count
        master_out = ma.masked_array(master / master_count, mask=master_count == 0)
        imputed = ma.masked_array(era5land_t2m_day.filled(0) + master_out.filled(0),
                                  mask=(master_out.mask * era5land_t2m_day.mask))
        imputed = imputed.reshape((1, 951, 951))
        imputed_out = np.vstack((imputed_out, imputed))
    #         gen_tiff(imputed[0],era5land_lat,era5land_lon,'final_%s_%s_%s.tif'%(year, Q, era5_day))
    #         gen_tiff(era5land_t2m_day,era5land_lat,era5land_lon,'era5land_%s_%s_%s.tif'%(year, Q, era5_day))
    era5.close()
    era5land.close()
    imputed_out = imputed_out[1:, :, :]
    imputed_out = ma.masked_array(imputed_out, imputed_out == 0)
    file_name = r'C:\Users\maria\Documents\work\Berkeley\data\imputation\Results_v2\%s.nc' % era5_land_nc.split('\\')[-1][9:-3]
    write_nc(file_name, era5land_lat, era5land_lon, era5_day_out, imputed_out)



