# Trailscan QGIS-Plugin

TrailScan is a Deep Learning Model which is trained to segment skid trails (syn. logging trails)  from Airborne Laser Scanning (ALS) data in Central European mixed and deciduous forests. ALS data is derived by planes, helicopters or unmanned aerial vehicles (UAV) carrying a LiDAR sensor. The 2-step TrailScan process is:

1. TrailScan Preprocessing: The ALS point cloud in .laz or .las data format is opened in QGIS. With the TrailScan preprocessing tool, the point cloud is converted into a 4-band georeferenced raster image. The 4 bands represent a Digital Terrain Model (DTM) in band 1, a Canopy Height Model (CHM) in band 2, a Micro Relief Model (MRM) in band 3 and a vegetation density index (VDI) in band 4. All values are normalized to the range of 0 to 1, therefore the name of the output file is "Normalized".

2. TrailScan Inference: The Normalized File gets processed with the TrailScan model. The output file is the prediction of the model that has a raster image format with values in the range of 0 to 1. Pixels with a value of 0 do not represent a skid trail. The higher the value, the higher the probability that a skid trail was found. 

## Requirements 

### Software

- QGIS 3.40 or newer
- Python packages laspy[lazrs,laszip], rasterio, onnxruntime, scipy

### LiDAR data

The data should be classified using the standard classes for point cloud data according to the LAS/LAZ file specifications and have a valid georeference. It is necessary to have classified ground points for the processes; if missing, you can achieve it using [PDAL](https://pdal.io/).

### Installation of Python packages

#### Linux 

    pip install laspy[lazrs,laszip] rasterio onnxruntime scipy

#### Windows

Open the Python console of QGIS and enter 

    !pip install laspy[lazrs,laszip] rasterio onnxruntime scipy

Please note that under windows each QGIS has it's own python, so you may have to reinstall the packages