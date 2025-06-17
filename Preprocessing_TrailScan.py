import json
import subprocess
import numpy as np
from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
import laspy
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling

# === Input / Output (r for reversed windows-backslash) ===
input_laz = r"F:\TrailScan\02_Daten\01_BB_Neuzelle\Neuzelle_Test.laz"
output_dtm = r"F:\TrailScan\02_Daten\01_BB_Neuzelle\Neuzelle_DTM.tif"
output_chm = r"F:\TrailScan\02_Daten\01_BB_Neuzelle\Neuzelle_CHM.tif"
output_lrm = r"F:\TrailScan\02_Daten\01_BB_Neuzelle\Neuzelle_LRM.tif"
output_vdi = r"F:\TrailScan\02_Daten\01_BB_Neuzelle\Neuzelle_VDI.tif"
output_raster = r"F:\TrailScan\02_Daten\01_BB_Neuzelle\Neuzelle__normalized.tif"
resolution = 0.38  # Resolution (raster width)

# === Function: Calculate uniform extent and transform ===
def calculate_extent_and_transform(input_laz, resolution):
    las = laspy.read(input_laz)
    x_min, x_max = np.min(las.x), np.max(las.x)
    y_min, y_max = np.min(las.y), np.max(las.y)

    # Calculate number of pixels
    width = int(np.ceil((x_max - x_min) / resolution))
    height = int(np.ceil((y_max - y_min) / resolution))

    # Transform object
    transform = from_origin(x_min, y_max, resolution, resolution)

    return transform, width, height

# === Function: Execute PDAL pipeline ===
def run_pdal_pipeline(pipeline, pipeline_file):
    with open(pipeline_file, "w") as f:
        json.dump(pipeline, f)

    try:
        subprocess.run(["pdal", "pipeline", pipeline_file], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"PDAL pipeline failed: {e}")

# === Function: Create DTM ===
def create_dtm(input_laz, output_dtm, resolution, transform, width, height):
    pipeline = [
        {"type": "readers.las", "filename": input_laz},
        {"type": "filters.outlier", "method": "statistical", "mean_k": 8, "multiplier": 3.0},
        # Filter for ground points and create DTM
        {"type": "filters.expression", "expression": "Classification == 2"},
        {
            "type": "writers.gdal",
            "filename": output_dtm,
            "resolution": resolution,
            "output_type": "idw",
            "power": 2.0,  # Increased power for better interpolation
            "window_size": 6,  # Adjusted window size
            "data_type": "float32",
            "nodata": -9999,
            "dimension": "Z"  # Explicitly specify we want to interpolate Z values
        }
    ]
    run_pdal_pipeline(pipeline, "dtm_pipeline.json")

# === Function: Normalize point cloud heights ===
def normalize_height(x, y, z, dtm_array, transform):
    from rasterio.transform import rowcol
    rows, cols = rowcol(transform, x, y)
    z_normalized = z - dtm_array[rows, cols]
    return z_normalized

# === Function: Calculate CHM ===
def calculate_chm(x, y, z, resolution, width, height):
    """Calculate Canopy Height Model (CHM) from normalized point cloud data.
    
    Args:
        x, y, z: Point cloud coordinates and heights
        resolution: Raster resolution
        width, height: Output raster dimensions
    """
    chm_stat, _, _, _ = binned_statistic_2d(
        x, y, z, statistic='max', bins=[width, height]
    )
    chm_stat_rotated = np.flipud(np.fliplr(np.rot90(chm_stat, k=3)))
    return np.nan_to_num(chm_stat_rotated, nan=0)

# === Calculate LRM ===
def calculate_lrm(dtm_array, sigma=5, mode='reflect', truncate=3.0, clip_range=(-2, 2)):
    """Calculate Local Relief Model (LRM) from DTM.
    
    Args:
        dtm_array: Digital Terrain Model array
        sigma (float): Standard deviation for Gaussian kernel. Default is 5.
            Smaller values create more contrast but may show more noise.
        mode (str): The mode parameter determines how the array borders are handled.
            Options are 'constant', 'reflect', 'nearest', 'mirror', 'wrap'.
            Default is 'reflect' for better edge handling.
        truncate (float): Truncate the filter at this many standard deviations.
            Default is 3.0 for wider filter influence.
        clip_range (tuple): Range to clip the output values to. Default is (-2, 2).
            Wider range preserves more contrast.
    """
    # Create a mask for valid data (non-NaN values)
    valid_mask = ~np.isnan(dtm_array)
    
    # Replace NoData-values with 0 for processing
    dtm_processed = np.nan_to_num(dtm_array, nan=0)

    # Smoothen DTM with gaussian_filter
    dtm_smoothed = gaussian_filter(dtm_processed, sigma=sigma, mode=mode, truncate=truncate)

    # Subtract the smoothed DTM from original DTM
    lrm = dtm_processed - dtm_smoothed

    # Clip values to specified range
    lrm = np.clip(lrm, clip_range[0], clip_range[1])

    # Restore NoData values
    lrm[~valid_mask] = np.nan

    return lrm

# === Calculate VDI ===
def calculate_vdi(x, y, z, resolution, width, height):
    """Calculate Vegetation Density Index (VDI) from normalized point cloud data.
    
    Args:
        x, y, z: Point cloud coordinates and heights
        resolution: Raster resolution
        width, height: Output raster dimensions
    """
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    # Filter points below 12m height
    mask = (z <= 12)
    z_sel = z[mask]
    x_sel = x[mask]
    y_sel = y[mask]

    if x_sel.size == 0 or y_sel.size == 0 or z_sel.size == 0:
        print("Warning: No points found for VDI calculation.")
        return np.zeros((height, width))

    # Filter points below 0.8m height
    z_low = z_sel[z_sel <= 0.8]
    x_low = x_sel[z_sel <= 0.8]
    y_low = y_sel[z_sel <= 0.8]

    # Calculate total and low vegetation density
    total_density, _, _, _ = binned_statistic_2d(
        x_sel, y_sel, None, statistic='count', bins=[width, height]
    )
    low_density, _, _, _ = binned_statistic_2d(
        x_low, y_low, None, statistic='count', bins=[width, height]
    )

    # Calculate VDI ratio
    vdi = np.divide(
        low_density, total_density, out=np.full_like(low_density, 0), where=total_density != 0
    )

    # Rotate and interpolate missing values
    vdi_rotated = np.flipud(np.fliplr(np.rot90(vdi, k=3)))
    vdi_interpolated = interpolate_missing_values(vdi_rotated, method='nearest')
    return vdi_interpolated

# === Interpolate missing values ===
def interpolate_missing_values(grid, method='nearest'):
    x, y = np.indices(grid.shape)
    valid_mask = ~np.isnan(grid)
    points = np.array([x[valid_mask], y[valid_mask]]).T
    values = grid[valid_mask]

    interpolated_grid = griddata(
        points, values, (x, y), method=method, fill_value=0
    )
    return interpolated_grid

# === Function: Normalize by percentiles ===
def normalize_percentile(data, low=1, high=99, nodata_value=0):
    """Normalizes data by applying a percentile cut stretch bandwise.

    Args:
        data (np.ndarray): The data to be normalized
        low: The low percentile cut value (default: 1)
        high: The high percentile cut value (default: 99)
        nodata_value: The no data value (default: 0)
    """
    # Create mask for valid data
    datamask = data != nodata_value
    
    # Calculate percentiles only for valid data
    pmin = np.array([np.percentile(data[:, :, i][datamask[:, :, i]], q=low) for i in range(data.shape[-1])])
    pmax = np.array([np.percentile(data[:, :, i][datamask[:, :, i]], q=high) for i in range(data.shape[-1])])
    
    # Normalize and clip
    normalized_data = np.clip((data - pmin) / (pmax - pmin + 1E-10), 0, 1)
    
    # Set NoData values back to 0
    normalized_data[~datamask] = nodata_value
    
    return normalized_data

# === Function: Create single raster ===
def create_single_raster(data_array, transform, output_path, crs="EPSG:25832", nodata_value=0):
    """Create a single-band raster from a data array.
    
    Args:
        data_array: Input data array
        transform: Raster transform
        output_path: Output file path
        crs: Coordinate reference system
        nodata_value: NoData value to use
    """
    height, width = data_array.shape

    with rasterio.open(
        output_path, "w",
        driver="GTiff", height=height, width=width,
        count=1, dtype=data_array.dtype, crs=crs, transform=transform,
        nodata=nodata_value
    ) as dst:
        dst.write(data_array, 1)

# === Function: Create multiband raster ===
def create_multiband_raster(data_arrays, transform, output_path, crs="EPSG:25832", nodata_value=0):
    """Create a multi-band raster from multiple data arrays.
    
    Args:
        data_arrays: List of input data arrays
        transform: Raster transform
        output_path: Output file path
        crs: Coordinate reference system
        nodata_value: NoData value to use
    """
    num_bands = len(data_arrays)
    height, width = data_arrays[0].shape

    with rasterio.open(
        output_path, "w",
        driver="GTiff", height=height, width=width,
        count=num_bands, dtype=data_arrays[0].dtype, crs=crs, transform=transform,
        nodata=nodata_value
    ) as dst:
        for i, data_array in enumerate(data_arrays, start=1):
            dst.write(data_array, i)

# === Function: Normalize LRM values around zero ===
def normalize_lrm(lrm_array, low=1, high=99, nodata_value=0):
    """Normalizes LRM data by applying a symmetric normalization around zero.
    
    Args:
        lrm_array: The LRM data to be normalized
        low: The low percentile cut value (default: 1)
        high: The high percentile cut value (default: 99)
        nodata_value: The no data value (default: 0)
    """
    # Create mask for valid data
    datamask = lrm_array != nodata_value
    
    # Calculate absolute values for percentile calculation
    abs_values = np.abs(lrm_array[datamask])
    max_abs = np.percentile(abs_values, q=high)
    
    # Normalize symmetrically around zero
    normalized_lrm = np.zeros_like(lrm_array)
    normalized_lrm[datamask] = np.clip(lrm_array[datamask] / (max_abs + 1E-10), -1, 1)
    
    # Scale to 0-1 range
    normalized_lrm[datamask] = (normalized_lrm[datamask] + 1) / 2
    
    # Set NoData values back to 0
    normalized_lrm[~datamask] = nodata_value
    
    return normalized_lrm

# === Main logic ===
if __name__ == "__main__":
    # Step 1: Calculate uniform extent and transform
    transform, width, height = calculate_extent_and_transform(input_laz, resolution)

    # Step 2: Create DTM
    create_dtm(input_laz, output_dtm, resolution, transform, width, height)

    # Step 3: Load DTM
    with rasterio.open(output_dtm) as src:
        dtm_array = src.read(1)
        nodata_value = src.nodata

    # Step 4: Load point cloud and normalize
    las = laspy.read(input_laz)
    x, y, z = las.x, las.y, las.z
    z_normalized = normalize_height(x, y, z, dtm_array, transform)

    # Step 5: Create and save CHM
    chm_array = calculate_chm(x, y, z_normalized, resolution, width, height)
    create_single_raster(chm_array, transform, output_chm, nodata_value=nodata_value)

    # Step 6: Calculate and save LRM
    lrm_array = calculate_lrm(dtm_array, sigma=5, mode='reflect', truncate=3.0, clip_range=(-2, 2))
    lrm_normalized = normalize_lrm(lrm_array, nodata_value=nodata_value)
    create_single_raster(lrm_normalized, transform, output_lrm, nodata_value=nodata_value)

    # Step 7: Calculate and save VDI
    vdi_array = calculate_vdi(x, y, z_normalized, resolution, width, height)
    create_single_raster(vdi_array, transform, output_vdi, nodata_value=nodata_value)

    # Step 8: Combine arrays and normalize
    combined_array = np.stack([dtm_array, chm_array, lrm_array, vdi_array], axis=2)
    normalized_array = normalize_percentile(combined_array, nodata_value=nodata_value)

    # Step 9: Save normalized raster
    create_multiband_raster(
        [normalized_array[:,:,i] for i in range(4)], 
        transform, 
        output_raster,
        nodata_value=nodata_value
    )

    # Cleanup
    import os
    os.remove("dtm_pipeline.json")

    print("All rasters successfully created!")
