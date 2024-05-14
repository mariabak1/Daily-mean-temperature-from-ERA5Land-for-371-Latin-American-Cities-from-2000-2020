
# Computing daily mean temperature from ERA5Land for 371 Latin American Cities from 2000-2015

Accociations between heat and mortality are one of the main interest of researchers involved in [SALURBAL-Climate](https://drexel.edu/lac/salurbal/overview/). SALURBAL-Climate uses temperature data from the ERA5Land reanalysis dataset. Here we explain the processing of temperature data. 

## 1. Data Imputation

ERA5Land data neglects pixels that have more than 50% water. However, many cities across the world are situated next to the ocean. To prevent missing (incomplete) temperature data from coastal cities, our team interpolated data from ERA5 (at a 31km x 31km resolution) and imputed it at the ERA5Land 9km x 9km resolution, filling the gaps from those "missing" pixels. This ensured that the ambient 2 meter temperature from ERA5Land covered the entire spatial extent of SALURBAL cities.

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


## 2. Calculating areal-level temperature
We used imputed ERA5Land data from the previous step to compute mean daily temperature for 2000-2020 for different types of SALURBAL geographies -- cities (AD), sub-cities (L2), and cities' urban extent (UX). We provide area-weighted averages (accounting for partial ERA5Land pixels) for each spatial unit as well as averages further weighted by population. Population-weighting is done using 1x1 km [WorldPop data](https://hub.worldpop.org/geodata/listing?id=64). 
For a spatial unit, its daily mean temperature is calculated as:

<div style="text-align:center">

$$
T_d = \frac{{\sum_{i=1}^m w_i T_{i,d}}}{{\sum_{i=1}^m w_i}}
$$

</div>

where:
- T_(i,d) is the daily mean temperature of day d in grid cell i, T_d is the area-level weighted average of daily mean temperature on day d,
-  $$ w_i $$  is a weighting factor, which equals to: w_i=〖area〗_i for area weighted values,
w_i=〖area〗_i×〖population(GUF)〗_i          for area and population (GUF) weighted values


$$ w_i = \text{area}_i \quad \text{for area weighted values,} $$

$$ w_i = \text{area}_i \times \text{population(GUF)}_i \quad \text $$ {for area and population (GUF) weighted values} 

- $$\(w_i = \text{area}_i\) where \(\text{area}_i\) $$ is the area of the overlap between grid cell \(i\) and the spatial unit, for area weighted values. 

- $$ \(w_i = \text{area}_i \times \text{population(GUF)}_i\) where \(\text{area}_i\) $$ is the area of the overlap between grid cell \(i\) and the spatial unit, and \(\text{population(GUF)}_i\) is the population or GUF urban footprint area of grid cell \(i\), for area and population (GUF) weighted values.

- \(w_i = \text{area}_i\) where \(\text{area}_i\) is the area of the overlap between grid cell \(i\) and the spatial unit, for area weighted values.

- \(w_i = \text{area}_i \times \text{population(GUF)}_i\) where \(\text{area}_i\) is the area of the overlap between grid cell \(i\) and the spatial unit, and \(\text{population(GUF)}_i\) is the population or GUF urban footprint area of grid cell \(i\), for area and population (GUF) weighted values.




### Access to raw data:
- [ERA5 hourly data on single levels](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview)
- [WorldPop data](https://www.worldpop.org/project/categories?id=3)
- [Global Urban Footprint data](https://drive.google.com/drive/folders/1_NM6c_SDAqb0LAOXt8LpbTT7eIL3HgAY)

### Access to imputed data:
- [ERA5Land imputed data](https://drive.google.com/drive/folders/1Eps9MGcVt_7Yn32Yboub3FVZZnQVrD14?usp=sharing)

### Access to final data:
- [L1 AD and UX data](https://drive.google.com/file/d/1Is1w0Oq5boAW6SlGlMWNP3C1IeiRey8R/view?usp=sharing)
- [L2 data](https://drive.google.com/file/d/1oa72qk_2zzbePTr9kCKzKs4DxhZl7I5Z/view?usp=sharing)

---


## Calculting areal-level temperature 
Daily mean temperature at the city and sub-city levels from 1996 to 2015. We provide area-weighted averages for each spatial unit as well as averages further weighted by population, using 100m x 100m WorldPop data for 2010. Since population data is not accurate for Panama and Peru, for cities in these two countries we weight temperature by urban footprint data (Global Urban Footprint). 
For a spatial unit, its daily mean temperature is calculated as:
T_(i,d)=(∑_(t=1)^n▒T_(t,i,d) )/n
T_d=(∑_(i=1)^m▒〖w_i T〗_(i,d) )/(∑_(i=1)^m▒w_i )
Where T_(t,i,d) is the temperature of hour t in grid cell i on day d, T_(i,d) is the daily mean temperature of day d in grid cell i, T_d is the area-level weighted average of daily mean temperature on day d, w_i is a weighting factor, which equals to:
w_i=〖area〗_i                                                   for area weighted values,
w_i=〖area〗_i×〖population(GUF)〗_i          for area and population (GUF) weighted values
where 〖area〗_i is the area of the overlap between grid cell i and the spatial unit, 〖population(GUF)〗_i is the population or GUF urban footprint area of grid cell i



**Notes**:  
- The final tables of population-weighted mean daily temperature are L1AD_UX_96_15.csv and L2_96_15.csv. 
---

**Codebook for [L1AD and UX data](https://drive.google.com/file/d/1Is1w0Oq5boAW6SlGlMWNP3C1IeiRey8R/view?usp=sharing):**  
- SALID1: City ID. (6 digits)
- ADtemp_pw: Population weighted temperature mean at L1AD level (city-level). 
- ADtemp_x: Unweighted temperature mean at L1AD level (city-level). 
- UXtemp_pw:Population weighted temperature mean at L1UX level (urban extent). 
- UXtemp_x:  Unweighted temperature mean at L1UX level (urban extent). 
- date: year-month-day.

Preview *L1AD_UX_96_15.csv*:  

<img src="scripts/L1_preview.png" align="center" width="60%">

**Codebook for [L2 data](https://drive.google.com/file/d/1oa72qk_2zzbePTr9kCKzKs4DxhZl7I5Z/view?usp=sharing):**  
- SALID2: Sub-city ID (8 digits). 
- L2temp_pw: Population weighted temperature mean at L2 level (sub-city). 
- L2temp_x: Unweighted temperature mean at L2 level (sub-city). 
- date: year-month-day. 

Preview *L2_96_15.csv*:  

<img src="scripts/L2_preview.png" align="center" width="40%">

**Contact:** 
- Yang Ju (yangju90@berkeley.edu)
- Irene Farah (irenef@berkeley.edu)
- Maryia Bakhtsiyarava (mariab@berkeley.edu)

