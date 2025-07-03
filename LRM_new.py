import rasterio
import numpy as np
from scipy.ndimage import gaussian_filter
import os

dem_path = "C:/Users/tanja/Projects/TrailScan_Plugin/Testdaten/2Solling_DTM.tif"
output_path = "C:/Users/tanja/Projects/TrailScan_Plugin/Testdaten/2Solling_LRM_new.tif"


with rasterio.open(dem_path) as src:
    dem = src.read(1)
    nodata = src.nodata

# LRM berechnen
def lrm(dem, radius, nodata_value, filter_type="gaussian", gaussian_sigma=5):
    """
    Berechnet das Local Relief Model (LRM) aus einem DEM.
    
    Parameters:
    - dem: numpy array des DEM
    - radius: Radius für die Filterung
    - nodata_value: Nodata Wert
    - filter_type: Typ des Filters ("gaussian")
    - gaussian_sigma: Sigma für den Gauß-Filter
    
    Returns:
    - numpy array mit dem LRM
    """
    # Handle NoData values
    dem = np.where(dem == nodata_value, np.nan, dem)
    
    # Apply Gaussian filter with specified parameters
    smoothed_dem = gaussian_filter(dem, sigma=gaussian_sigma, mode='reflect', truncate=3.0)
    
    # Calculate LRM
    lrm = dem - smoothed_dem
    
    # Replace NaNs with nodata
    # For uint8 output, we'll use 0 as nodata value
    lrm = np.where(np.isnan(lrm), 0, lrm)
    
    return lrm

# LRM berechnen
lrm_image = lrm(dem, radius=9, nodata_value=nodata, filter_type="gaussian", gaussian_sigma=3)

# Normalize LRM values to 0-255 range
print("Normalizing LRM values to 0-255 range...")
# Get valid data mask
valid_mask = ~np.isnan(lrm_image)
if np.any(valid_mask):
    # Get min and max values from valid data
    min_val = np.nanmin(lrm_image)
    max_val = np.nanmax(lrm_image)
    print(f"Original LRM value range: {min_val} to {max_val}")
    
    # Normalize to 0-255 range
    lrm_image[valid_mask] = ((lrm_image[valid_mask] - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    print("LRM normalized to 0-255 range")
else:
    print("No valid data found for normalization")

# Ergebnis speichern
print(f"Saving LRM to: {output_path}")
try:
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=lrm_image.shape[0],
        width=lrm_image.shape[1],
        count=1,
        dtype=rasterio.uint8,
        crs=src.crs,
        transform=src.transform,
        nodata=0
    ) as dst:
        dst.write(lrm_image, 1)
    print("LRM saved successfully!")
except Exception as e:
    print(f"Error saving LRM: {str(e)}")
