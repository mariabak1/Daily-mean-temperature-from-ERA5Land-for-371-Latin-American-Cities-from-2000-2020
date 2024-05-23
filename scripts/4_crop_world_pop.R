# 4/11/2024
# Author: Maryia Bakhtsiyarava
# This script crops and resamples WorldPop 1x1 km files so that the WorldPop cells are aligned with (imputed) ERA5Land.
# The script loops through the WorldPop files, resamples them and stores in a directory of your choice.
# The resampled WorldPop files are then used for pop-weighting in the next step.
##### USERS ONLY NEED TO CHANGE INFO IN LINES 23-32 #####
# WorldPop was downloaded from here https://hub.worldpop.org/geodata/listing?id=64

# load packages

library(raster)
library(dplyr)
library(ncdf4)
library(data.table)
library(terra)
library(tidyr) # needed to work with dates
library(lubridate)
library(stringr) # work with file names


################################################
############### CHANGE THINGS HERE #############
################################################

# provide a sample of temperature data for resampling
temp="2016_Q1.nc" # path to an era5land file (can be any year)
era5land=stack(temp) # read in the file

# World pop data
pop_files_path = "C:/pop files/"  # folder with worldpop data
pop_files = list.files(pop_files_path, pattern = "\\.tif$", full.names = TRUE)

# specify directory to store resampled files
out_directory = "C:/Users/cropped_worldpop/"   # for example

###################################################
############ END OF CHANGES #######################
###################################################




for (c in pop_files) {
  year=regmatches(c, regexpr("\\d{4}", c)) # extract year from a worldpop file
  worldpop=raster(c) # read in worldpop file
  print(worldpop)
  
  cellfactor = res(era5land)[1] / res(worldpop)[1] #ratio of cell sizes for aggregation
  
  # since the resolutions of worldpop and era5land are different, we aggregate the worldpop to the same cell size as era5land
  extent=extent(-122.05, -22.95, -62.05, 37.05) # add a few (2) degrees to era5land extent to ensure compatible extents
  worldpop_cropped = crop(worldpop, extent) # crop the global worldpop to era5land's extent
  worldpop_cropped=terra::aggregate(worldpop_cropped, fact=cellfactor, fun="sum") # aggregate population counts by sum
  worldpop_cropped=resample(worldpop_cropped, era5land) # resample so that cells align

  # save the resampled worldpop files # chose your locations; the files will be called e.g., "wp2020.tif"
  fname=paste(out_directory,"wp", year, '.tif', sep = "")
  writeRaster(worldpop_cropped, fname)
    
}
