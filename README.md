
# Computing daily mean temperature from ERA5Land for 371 Latin American Cities from 2000-2015

Accociations between heat and mortality are one of the main interests of researchers involved in [SALURBAL-Climate](https://drexel.edu/lac/salurbal/overview/). SALURBAL-Climate uses temperature data from the ERA5Land reanalysis dataset. Here we explain the processing of temperature data. 
## 1. Data Download

Download temperature data from [Copernicus](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=overview). ERA5Land compiles land surface temperature data, while ERA5 compiles surface temperature data for land and water. The boundaries of coastal cities often contain water, so only using ERA5Land means that we’d miss a portion of temperature data for these cities. Therefore, we download two datasets and explain the imputation below. You should get .nc files as the output.

[ERA5Land dataset](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=overview)

 - Relevant variables: 2m temperature

The ERA5Land data should be downloaded on a quarterly basis. For example, era5land_2016_Q1, 2016_Q2, etc. This can be tedious to do using Copernicus’ UI, so you can also download using their [API](scripts/1_download_era5land.ipynb?short_path=27668ea). The download script contains details about the geographic extent for Latin American cities we're intested in downloading.

[ERA5 dataset](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview)

- Relevant variables: 2m temperature

The ERA5 data should be downloaded on a yearly basis. For example, era5_2016, era5_2017, etc. We downloaded it manually without API because it was easy.

For both datasets, you can specify what geographical region, years, months, variables, etc. you’re interested in getting.


## 2. Data Imputation

ERA5Land data neglects pixels that have more than 50% water. However, many cities across the world are situated next to the ocean. To prevent missing (incomplete) temperature data from coastal cities, our team interpolated data from ERA5 (at a 31km x 31km resolution) and imputed it at the ERA5Land 9km x 9km resolution, filling the gaps from those "missing" pixels. This ensured that the ambient 2 meter temperature from ERA5Land covered the entire spatial extent of SALURBAL cities. 

Here is an example of missing pixels for Pisco, Peru:

![Example of missing pixels for Pisco, Peru](https://github.com/mariabak1/Daily-mean-temperature-from-ERA5Land-for-371-Latin-American-Cities-from-2000-2020/assets/67489014/b3aa9255-ea95-480d-83b2-b50b801ee846)


To impute the missing values, we built the following model for each day and geography (L1AD) with missing ERA5land pixels: 
<div style="text-align:center">

$$
\text{ERA5Land} = f(X) + \epsilon
$$

</div>

where: 
- X is a vector including resampled ERA5 temperature from 31 km resolution to 9 km resolution with cubic resampling, absolute elevation (9 km resolution), relative elevation (elevation difference of a 9x9 km pixel and its surroundings), and aspect (9 km resolution);  
- f(X) is a function that uses X to regress ERA5land temperature. Here we used random forest regression.  
- ϵ is the residual, or ERA5Land - f(X), which we further modeled with kriging spatial interpolation.

We ran this model by each day and each geographic unit (L1AD) containing missing ERA5Land missing pixels. In each geographic unit, we included all ERA5land pixels and pixels within a 15-pixel buffer from the boundary to have enough samples to build the model above. To avoid overfitting, we used cross-validation to tune the parameters for both random forest regression and kriging spatial interpolation. Finally, we used the resulting model to impute the missing values.  

**Note**: In two cases we did not include kriging spatial interpolation in the imputation:
1. If adding kriging spatial interpolation led to worse model fit when compared with using random forest regression alone, in where we had both ERA5 and ERA5land coverage;
2. If kriging spatial interpolation produced large values, which we seldomly found for the missing pixels. We decided the threshold to be 1 degree in absolute value. We chose this threshold as we observed the model residuals from using random forest alone were less than 1. Since kriging spatial interpolation was meant to further reduce these residuals, we considered kriging values greater than 1 to be anomalies and thus abandoned. 


[Code to perform imputation](scripts/2_era5land_imputation.py). The script takes in the era5land_year_quarter.nc and era5.nc files as parameters, among other things like .tif files for the city boundaries, etc.

[Code to compute the percentage of imputed pixels by level of geography](scripts/3_compute_pct_imputed.ipynb)

## 3. Calculating areal-level temperature
We used imputed ERA5Land data from the previous step to compute mean daily temperature for 2000-2020 for different types of SALURBAL geographies -- cities (AD), sub-cities (L2), and cities' urban extent (UX). We provide area-weighted averages (accounting for partial ERA5Land pixels) for each spatial unit as well as averages further weighted by population. Population-weighting is done using 1x1 km [WorldPop data](https://hub.worldpop.org/geodata/listing?id=64). 
For a spatial unit, its daily mean temperature is calculated as:

<div style="text-align:center">

$$
T_d = \frac{{\sum_{i=1}^m w_i T_{i,d}}}{{\sum_{i=1}^m w_i}}
$$

</div>

where:
- <code>T<sub>i,d</sub></code> is the daily mean temperature of day d in grid cell i,
- <code>T<sub>d</sub></code> is the area-level weighted average of daily mean temperature on day d,
- <code>w<sub>i</sub></code> is a weighting factor, which equals to:
  
$$ w_i = \text{area}_i \quad \text{for area weighted values,} $$

$$
w_i = \text{area}_i \times \text{population(WorldPop)}_i
$$

where <code>area<sub>i</sub></code> is the area of the overlap between grid cell i and the spatial unit; <code>population(WorldPop)<sub>i</sub></code> is the WorldPop population of grid cell i.



Code to obtain mean daily temperature
 - [First run this script to resample WorldPop](scripts/4_crop_world_pop.R)
 - [Run this code to compute population-weighted (and unweigted) mean daily temperature](scripts/5_era5land_popweights.R)




## Access to raw data:
- [ERA5Land](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=overview)
- [ERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview) 
- [WorldPop data](https://hub.worldpop.org/geodata/listing?id=64)

## For access to processed data contact mariab@berkeley.edu


**Codebook for [L1AD and UX data]**  
- SALID1: City ID. (6 digits)
- L1ADtemp_pw: Population weighted temperature mean at L1AD level (city-level). 
- L1ADtemp_x: Unweighted temperature mean at L1AD level (city-level). 
- L1UXtemp_pw:Population weighted temperature mean at L1UX level (urban extent). 
- L1UXtemp_x:  Unweighted temperature mean at L1UX level (urban extent). 
- date: year-month-day.

L1AD/UX Dataset Preview: 

<img src="scripts/L1_preview.png" align="center" width="60%">

**Codebook for [L2 data]**  
- SALID2: Sub-city ID (8 digits). 
- L2temp_pw: Population weighted temperature mean at L2 level (sub-city). 
- L2temp_x: Unweighted temperature mean at L2 level (sub-city). 
- date: year-month-day. 

L2 Dataset Preview:

<img src="scripts/L2_preview.png" align="center" width="40%">

**Contact:** 
- Maryia Bakhtsiyarava (mariab@berkeley.edu)
- Brian Xi (brianx@berkeley.edu)

