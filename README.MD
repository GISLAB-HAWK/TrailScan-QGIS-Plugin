# TrailScan

**TrailScan** is a deep learning model designed to automatically detect and segment **skid trails (logging trails)** from **Airborne Laser Scanning (ALS)** data in Central European mixed and deciduous forests.  

ALS data is collected by aircraft, helicopters, or unmanned aerial vehicles (UAVs) equipped with LiDAR sensors. The TrailScan workflow consists of two main steps:

---

## 1. TrailScan Preprocessing

- **Input:** ALS point clouds in `.laz` or `.las` format are opened directly in QGIS.  
- The preprocessing tool converts the point cloud into a **4-band georeferenced raster image**:
  - **Band 1:** Digital Terrain Model (DTM)  
  - **Band 2:** Canopy Height Model (CHM)  
  - **Band 3:** Micro-Relief Model (MRM)  
  - **Band 4:** Vegetation Density Index (VDI)  
- All raster values are **normalized to a range between 0 and 1**, which is why the output is named the *Normalized File*.  

---

## 2. TrailScan Inference

- **Input:** the *Normalized File* generated in step 1.  
- Add the path to the `trailscan_model.onnx` file.  
- The TrailScan model processes the *Normalized File* and produces a **Trailmap**.  

**Trailmap** is a prediction raster with values between 0 and 1:  
- `0` = no skid trail  
- `>0` = probability of a skid trail (the higher the value, the more likely a trail is present).  

---

# Requirements

![QGIS](https://img.shields.io/badge/QGIS-3.40%2B-green?logo=qgis)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![ONNX](https://img.shields.io/badge/ONNX-runtime-orange?logo=onnx)
![License](https://img.shields.io/badge/License-GPL--2.0-red.svg)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows%20%7C%20macOS-lightgrey)

---

## Software

- **QGIS** 3.40 or newer  
- **Python packages:**
  - `laspy[lazrs,laszip]`
  - `rasterio`
  - `onnxruntime`
  - `scipy`

---

## Installation of Python Packages

- **Linux:** 

  ```bash pip install laspy[lazrs,laszip] rasterio onnxruntime scipy ``` 

- **Windows:**  
  Open the **OSGeo4W Shell** or the **Python Console of QGIS** and run:  

  ```bash
  !pip install laspy[lazrs,laszip] rasterio onnxruntime scipy
  > **Note:** On Windows, each QGIS installation has its own Python environment. You may need to reinstall the packages after updating or changing your QGIS version. 

- **macOS:** 

  ```bash pip3 install laspy[lazrs,laszip] rasterio onnxruntime scipy ``` 
  > **Tip:** Depending on your macOS setup, you may need a dedicated Python environment (e.g., via conda or venv) to ensure compatibility with QGIS. 

---
## ALS LiDAR Data: 

- Input data must be provided in LAS/LAZ format with a valid georeference. 
- Point clouds should follow the standard classification scheme as defined in the LAS/LAZ specifications. Classified ground points (class 2) are required for the preprocessing step. If ground classification is missing, it can be generated using PDAL or equivalent tools. 
- Data should have sufficient point density for reliable DTM, CHM, MRM, and VDI derivation (minimum 6 points per m², recommended maximum 20 points per m²). 


---
## Hardware Recommendations: 

- CPU: Multi-core processor (Intel i7/i9 or AMD Ryzen recommended). 
- RAM: Minimum 16 GB (32 GB or more recommended for large point clouds).  
- GPU: Not required, but ONNX Runtime can optionally leverage GPU acceleration if supported drivers are installed.
