# Trailscan QGIS-Plugin

Detection of trails in LiDAR data.

## Requirements 

- QGIS 3.40 or newer
- Python packages laspy[lazrs,laszip], rasterio, onnxruntime, scipy

### Installation of Python packages

#### Linux 

    pip install laspy[lazrs,laszip] rasterio onnxruntime scipy

#### Windows

Open the Python console of QGIS and enter 

    !pip install laspy[lazrs,laszip] rasterio onnxruntime scipy

Please note that under windows each QGIS has it's own python, so you may have to reinstall the packages