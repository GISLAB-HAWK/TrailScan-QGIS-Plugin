# Trailscan QGIS-Plugin

Detection of trails in LiDAR data.

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