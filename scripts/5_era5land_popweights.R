# This script imports imputed era5land, worldpop data and city/sub-city boundaries and 
# computes population-weighted and un-weighted mean daily temperature in a city/sub-city. The script also weights by era5land pixel area to take into account cases when cities contain partial pixels
# author: Maryia Bakhtsiyarava mariab@berkeley.edu
# 4/11/2024

library(raster)
library(dplyr)
library(ncdf4)
library(data.table)
library(sf)
library(ggplot2)
library(exactextractr)
library(terra)
library(tidyr) # these two are needed to work with dates
library(lubridate)

################################################################################################
# For script user: You will need to 
#     1. indicate the locations of the folders/files (lines 34-46)
#     2. select L2 or L1 --> comment out line 49 or 50
#     3. specify the name and location of the csv file to store the results (line 119)
################################################################################################



########################
###### LOAD DATA #######
########################
# NOTE: Using nc_open from the ncdf4 package provides very detailed information about the netcdf temperature file, including
# units of measurement, variable names, etc. Using stack/brick functions does not provide as much detail but
# it's easier to work with the data when it's loaded using the stack/brick functions.
# So you could first load and explore the temperature data using the nc_open function but for the computations we use the file imported using the stack function

# load cities (ux, l1ad, or l2) # comment out the units you're not using
#cities=st_read("C:/level1_gcs_modify3.shp")
#cities=st_read("C:/level1_gcs_ux_modify4_20191127.shp")
cities=st_read("C:/level2_gcs_modify4.shp")

# World pop data
pop_files = "C:/Users/pop files/"  # folder with worldpop data

# Era5Land .nc era5land data (location of the IMPUTED era5land data)
folder_path = "C:/imputed era5land data"

# Get a list of files in the folder
nc_files = list.files(folder_path, pattern = "\\.nc$", full.names = TRUE)


#colname="SALID1" # select whether SALID1 or SALID2 (comment out whichever you're not using)
colname="SALID2"


###########################################################################################################
# Compute mean daily temperature by area: first without population weighting, then with population weights
###########################################################################################################

# Create a list to store results
DAT=list()

start_time <- Sys.time() # 
# Loop over each .nc file, load the worldpop file, and do the computation

for (c in nc_files) {
  era5land <- stack(c) # load era5land files
  print(era5land)
  print(paste("working on", c, sep=" "))
  
  #COMPUTE MEAN DAILY TEMP, UNWEIGHTED BY POPULATION #############
  datunw = exact_extract(rast(era5land), cities, fun = c("mean"), append_cols=colname)   # mean daily temp 
  
  datunw <- datunw %>% # convert to a long format
    pivot_longer(cols = contains("mean"),
                 names_to = "Date",
                 values_to = "value")
  
  datunw=datunw%>% # clean up
    mutate(date = ymd(sub("mean.X", "", Date)),
           ADtemp_x=value-273.15)%>% # convert to deg C
    dplyr::select(-value, -Date)
  
  
  # COMPUTE MEAN DAILY TEMP, WEIGHTED BY POPULATION #####
  # NOTE: If pop weighting is not needed, comment out the following lines
  
  year=sub(".*/([0-9]{4})_.*", "\\1", c) # get a year from the file name
  worldpopname=paste(pop_files, "wp", year, ".tif", sep="") # read in the corresponding year of worldpop
  worldpop=raster(worldpopname)
  print(worldpopname)
  print(paste("Population-weighting", year, sep=" "))
  
  datw = exact_extract(rast(era5land), cities, fun = c("weighted_mean"), weights=worldpop,  default_weight=0 ,append_cols=colname)   # mean daily temp 
  
  datw = datw %>% # convert to a long format
    pivot_longer(cols = contains("mean"),
                 names_to = "Date",
                 values_to = "value")
  # clean up
  datw=datw%>%
    mutate(date = ymd(sub("weighted_mean.X", "", Date)),
           ADtemp_pw=value-273.15)%>% # convert to deg C
    dplyr::select(-value, -Date)
  
  # combine unweighted and weighted
  dat=left_join(datunw, datw, by=c(colname, "date"))
  
  DAT[[c]] = dat
  
}
end_time <- Sys.time()
end_time - start_time


# combine into a dataframe
datr <- do.call(rbind, lapply(DAT, as.data.frame))
length(unique(datr$SALID1)) # ensure all cities are represented (N=371 L1s or L1UXs)
length(unique(datr$SALID2)) # ensure all cities are represented (N=1436 L2s)

# save files
write.csv(datr, "C:/l1ad_2016_2020.csv", row.names = F)




# extra: to access area- and population weights
wp_weights<- exact_extract(era5land, cities, weights=worldpop_cropped, include_cols="SALID1")   
# look at the pop and area weights for the first city
(wp_weights[[1]])


