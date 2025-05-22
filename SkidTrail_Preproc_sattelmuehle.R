##############################################################################
# Skid Trail Automation Pre-Processing from LAS Data, Tanja Kempen 21.12.2022
# Credits to: Prof. Dr. Paul Magdon, HAWK, Goettingen
# Credits to: Hans Fuchs, Georg-August-University, Goettingen
##############################################################################

# Initial Installation (if the following packages are not installed already) ---------------------------------
##install.packages("lidR", dependencies = TRUE)
##install.packages('spData')
##nstall.packages('rgdal')
##install.packages('spDataLarge', repos='https://nowosad.github.io/drat/', type='source')
##install.packages(sp)
##install.packages(raster)
##install.packages(sf)
##install.packages(terra)
##install.packages(tools)

# Load packages -----------------------------------------------------------------------------------------------

library(lidR)
#library(rgdal)
library(spData)
library(spDataLarge)
library(sp)
library(raster)
library(sf)
library(terra)
library(tools)

setwd("F:/TrailScan/02_Daten/09_Testdaten")

#################################################################
# 1. DTM (Digital Terrain Model)
#################################################################
# import LiDAR Data as "las" and set location name

las <- readLAS("Sattelm_Test_400m.laz")
##las2 <- readLAS("raw/las_32_615_5632_1_th_2020-2025.laz")
#las3 <- readLAS("raw/las_32_615_5633_1_th_2020-2025.laz")
#las4 <- readLAS("raw/las_32_616_5632_1_th_2020-2025.laz")
#las5 <- readLAS("raw/las_32_616_5633_1_th_2020-2025.laz")
#plot(las)
#merged <- rbind(las1, las2, las3, las4, las5)
location = "Sattelmuehle_Test"
# Abteilungspolygon einlesen
#abt = read_sf("Deselaers/catterfeld_UG_Polygon.gpkg")
# extract location name from file path and built file name for output
DTM_Location = paste("DTM_",location, sep="")
#clip las to abt
#las = clip_roi(merged, abt)
#plot(las)
epsg(las) = 25832
# remove point duplicates
las<-filter_duplicates(merged)
las
#plot(las)
#filter only ground classified data (class 2)
las_ground <- filter_poi(las, Classification == 2)  
las_ground
# Number of ground points
length(las_ground@data$X)
# Area
area(las_ground)
# point density/m2
d <- length(las_ground@data$X)/area(las_ground)
# point distance [m]
s <- sqrt(area(las_ground)/length(las_ground@data$X))
s
d
# interpolation of ground points by triangulated irregular network, res = 0.38,  tin())
# here, a grid width of 0.38 m was used
dtm = grid_terrain(las_ground, res = 0.38,  tin())
# save DTM to raster file
writeRaster(dtm, filename = DTM_Location, format = "GTiff", overwrite = T)

#################################################################
# 2. LRM (Local Relief Model)
#################################################################
#process the DTM in RVT_2.2.1 (https://www.zrc-sazu.si/en/rvt) 
#checkbox: "Simple Local Relief Model"
#radius for trend assessment (pixel): 10

#################################################################
# 3. CHM (Canopy Height Model)
#################################################################
las_canopy <- filter_poi(las, Classification == 2 | Classification == 20 )
# Interpolation of the ground returns using inverse distance weighted interpolation (IDW), 
# to generate a DTM with 1m resolution
dtm_chm = grid_terrain(las_ground, res = 1, knnidw(k=10, p=2), keep_lowest = FALSE)
# Create a normalized point cloud (removing the topography by substracting the terrain elevation from the Z-Value) -------------------------------------------------------------------------
las_normalized = normalize_height(las_canopy, dtm_chm)
epsg(las_normalized) = 25832
# Calculate a canopy height model (CHM)
# Method of Khosravipour et al. (2014) pitfree algorithm (computationally demanding: be patient)
# consists of several layers of triangulation at different elevations (thresholds in meters), gridsize = 0.25m.chm <- grid_canopy(las, res = 0.38, pitfree(thresholds = c(0,2,5,10,15,20,25,30,35), max_edge=c(0,1.5)))
chm <- grid_canopy(las_normalized, res = 0.38, pitfree(thresholds = c(0,2,5,10,15,20,25,30,35), max_edge=c(0,1.5)))
#plot(chm, col = height.colors(50),main='Canopy height model CHM')
#Built file name and save your chm to a raster file with:
CHM_Location = paste("CHM_",location, sep="")

writeRaster(chm, CHM_Location, format = "GTiff", overwrite = T)

#################################################################
# 4. VDI (Vegetation Density Index)
#################################################################
# Normalise the point cloud with the ground returns as provided by the vendor
# knnidw = Interpolation using k-nearest neighbour for normalizing height
nlas <- normalize_height(las_canopy, knnidw())
epsg(nlas) = 25832
# Calculate the metrics to describe the vegetation density
# This function calculates the proportion of returns that penetrate through a
# layer defined by the upper threshold [t_upper] (default=12m) and a lower threshold 
# [t_lower] (default=0.8m)
vegDen<-function(z,t_lower=0.8,t_upper=12){
  pts  <- z[z<=t_upper]
  pts_low <- pts[pts<=t_lower]
  prop = length(pts_low)/length(pts)
}
# To use customize metrics we need to pass them as a list
metrics_custom <- function(z) { # user defined function
  list(
    vegDensity <- vegDen(z))
}
# Run the calculation for the entire tile. Here we use a spatial resolution of  
# 2m to avoid many empty pixels
vegDensity <-pixel_metrics(nlas, ~metrics_custom(z=Z),res=2)
# Resample the vegetation density raster to the resolution of the dtm
dtm <- terra::rast(dtm)
vegDensity<-terra::resample(vegDensity,dtm)
# Export the vegetation density raster
VDI_Location = paste("VDI_",location, ".tif", sep="")
terra::writeRaster(vegDensity,
                   filename = VDI_Location, 
                   overwrite=T)



