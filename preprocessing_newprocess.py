"""
***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessing,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingMultiStepFeedback,
    QgsProcessingParameterPointCloudLayer,
    QgsProcessingParameterRasterDestination,
    QgsRasterLayer,
    Qgis,
    QgsCoordinateReferenceSystem
)
from qgis import processing
import numpy as np
from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
import laspy
import rasterio
from rasterio.transform import from_origin
from typing import Any
import json
import os
import subprocess
import itertools
import time

PIXEL_SIZE = 0.38

class TrailscanPreProcessingAlgorithm(QgsProcessingAlgorithm):
    """
    Algorithm for preprocessing point cloud data.
    """

    POINTCLOUD = "POINTCLOUD"
    OUTPUT_DTM = "OUTPUT_DTM"
    OUTPUT_CHM = "OUTPUT_CHM"
    OUTPUT_LRM = "OUTPUT_LRM"
    OUTPUT_VDI = "OUTPUT_VDI"
    OUTPUT_NORMALIZED = "OUTPUT_NORMALIZED"

    def name(self):
        return "preprocessing"

    def displayName(self):
        return "Preprocessing Point Cloud"

    def group(self):
        return "Trailscan"

    def groupId(self):
        return "trailscan"

    def shortHelpString(self):
        return "Preprocesses point cloud data to create DTM, CHM, LRM, and VDI rasters."

    def createInstance(self):
        return TrailscanPreProcessingAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterPointCloudLayer(
                name=self.POINTCLOUD,
                description="Input Point Cloud"
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                name=self.OUTPUT_DTM,
                description="DTM Output"
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                name=self.OUTPUT_CHM,
                description="CHM Output"
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                name=self.OUTPUT_LRM,
                description="LRM Output"
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                name=self.OUTPUT_VDI,
                description="VDI Output"
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                name=self.OUTPUT_NORMALIZED,
                description="Normalized Output"
            )
        )

    def processAlgorithm(self, parameters: dict[str, Any], context: QgsProcessingContext, feedback: QgsProcessingMultiStepFeedback) -> dict[str, Any]:
        counter = itertools.count(1)
        feedback = QgsProcessingMultiStepFeedback(9, feedback)

        # Get input parameters
        sourceCloud = self.parameterAsPointCloudLayer(parameters, self.POINTCLOUD, context)
        if sourceCloud is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.POINTCLOUD))

        input_laz = sourceCloud.dataProvider().dataSourceUri()
        
        # Get CRS from point cloud
        crs = sourceCloud.crs()
        if not crs.isValid():
            raise QgsProcessingException("Invalid CRS in input point cloud")
        
        crs_wkt = crs.toWkt()
        feedback.pushInfo(f"Using CRS: {crs.description()}")

        # Calculate extent and transform
        transform, width, height = calculate_extent_and_transform(input_laz, PIXEL_SIZE)
        feedback.setCurrentStep(next(counter))
        if feedback.isCanceled(): return {}

        # Create DTM using PDAL
        dtm_output = self.parameterAsOutputLayer(parameters, self.OUTPUT_DTM, context)
        
        # Run PDAL command directly to ensure filter is properly applied
        pdal_cmd = [
            "C:/OSGeo4W/apps/qgis/./pdal_wrench.exe",
            "to_raster_tin",
            "--input=" + input_laz,
            "--output=" + dtm_output,
            "--resolution=" + str(PIXEL_SIZE),
            "--tile-size=1000",
            "--filter=Classification == 2",
            "--threads=32"
        ]
        
        feedback.pushInfo(f"Running PDAL command: {' '.join(pdal_cmd)}")
        
        try:
            result = subprocess.run(pdal_cmd, check=True, capture_output=True, text=True)
            feedback.pushInfo(f"PDAL output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            feedback.pushInfo(f"PDAL command failed: {e.stderr}")
            raise QgsProcessingException(f"PDAL command failed: {e.stderr}")
        
        # Wait for file to be created
        max_wait = 10  # seconds
        wait_time = 0
        while not os.path.exists(dtm_output) and wait_time < max_wait:
            time.sleep(0.5)
            wait_time += 0.5
        
        if not os.path.exists(dtm_output):
            raise QgsProcessingException(f"DTM output file was not created: {dtm_output}")
        
        feedback.setCurrentStep(next(counter))
        if feedback.isCanceled(): return {}

        # Load DTM and get its dimensions
        try:
            with rasterio.open(dtm_output) as src:
                dtm_array = src.read(1)
                nodata_value = src.nodata
                height, width = dtm_array.shape
                transform = src.transform
        except Exception as e:
            raise QgsProcessingException(f"Failed to open DTM file: {str(e)}")

        # Load point cloud and normalize
        las = laspy.read(input_laz)
        x, y, z = las.x, las.y, las.z
        z_normalized = normalize_height(x, y, z, dtm_array, transform)

        # Create CHM with same dimensions as DTM
        chm_array = calculate_chm(x, y, z_normalized, PIXEL_SIZE, width, height)
        chm_output = self.parameterAsOutputLayer(parameters, self.OUTPUT_CHM, context)
        create_single_raster(chm_array, transform, chm_output, crs=crs_wkt, nodata_value=nodata_value)
        feedback.setCurrentStep(next(counter))
        if feedback.isCanceled(): return {}

        # Create LRM with same dimensions as DTM
        lrm_array = calculate_lrm(dtm_array, sigma=5, mode='nearest', truncate=3.0, clip_range=(-2, 2))
        lrm_output = self.parameterAsOutputLayer(parameters, self.OUTPUT_LRM, context)
        create_single_raster(lrm_array, transform, lrm_output, crs=crs_wkt, nodata_value=nodata_value)
        feedback.setCurrentStep(next(counter))
        if feedback.isCanceled(): return {}

        # Create VDI with same dimensions as DTM
        vdi_array = calculate_vdi(x, y, z_normalized, PIXEL_SIZE, width, height)
        vdi_output = self.parameterAsOutputLayer(parameters, self.OUTPUT_VDI, context)
        create_single_raster(vdi_array, transform, vdi_output, crs=crs_wkt, nodata_value=nodata_value)
        feedback.setCurrentStep(next(counter))
        if feedback.isCanceled(): return {}

        # Verify array shapes before stacking
        feedback.pushInfo(f"Array shapes - DTM: {dtm_array.shape}, CHM: {chm_array.shape}, LRM: {lrm_array.shape}, VDI: {vdi_array.shape}")
        
        # Ensure all arrays have the same shape
        target_shape = dtm_array.shape
        if chm_array.shape != target_shape:
            chm_array = np.resize(chm_array, target_shape)
        if lrm_array.shape != target_shape:
            lrm_array = np.resize(lrm_array, target_shape)
        if vdi_array.shape != target_shape:
            vdi_array = np.resize(vdi_array, target_shape)

        # Combine arrays and normalize
        combined_array = np.stack([dtm_array, chm_array, lrm_array, vdi_array], axis=2)
        normalized_array = normalize_percentile(combined_array, nodata_value=nodata_value)

        # Create normalized output
        normalized_output = self.parameterAsOutputLayer(parameters, self.OUTPUT_NORMALIZED, context)
        create_multiband_raster(
            [normalized_array[:,:,i] for i in range(4)], 
            transform, 
            normalized_output,
            crs=crs_wkt,
            nodata_value=nodata_value
        )
        feedback.setCurrentStep(next(counter))

        return {
            self.OUTPUT_DTM: dtm_output,
            self.OUTPUT_CHM: chm_output,
            self.OUTPUT_LRM: lrm_output,
            self.OUTPUT_VDI: vdi_output,
            self.OUTPUT_NORMALIZED: normalized_output
        }

# Helper functions
def calculate_extent_and_transform(input_laz, resolution):
    las = laspy.read(input_laz)
    x_min, x_max = np.min(las.x), np.max(las.x)
    y_min, y_max = np.min(las.y), np.max(las.y)

    width = int(np.ceil((x_max - x_min) / resolution))
    height = int(np.ceil((y_max - y_min) / resolution))

    transform = from_origin(x_min, y_max, resolution, resolution)

    return transform, width, height

def normalize_height(x, y, z, dtm_array, transform):
    from rasterio.transform import rowcol
    rows, cols = rowcol(transform, x, y)
    
    # Ensure indices are within bounds
    valid_rows = np.clip(rows, 0, dtm_array.shape[0] - 1)
    valid_cols = np.clip(cols, 0, dtm_array.shape[1] - 1)
    
    # Get DTM values for each point
    dtm_values = dtm_array[valid_rows, valid_cols]
    
    # Calculate normalized height
    z_normalized = z - dtm_values
    
    # Filter out unrealistic values
    z_normalized = np.clip(z_normalized, -0.5, 50)
    
    return z_normalized

def calculate_chm(x, y, z, resolution, width, height):
    # Filter out unrealistic values
    valid_mask = (z >= -0.5) & (z <= 50)  # Filter for realistic height values
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    z_valid = z[valid_mask]

    # Calculate CHM using binned statistics
    chm_stat, x_edges, y_edges, _ = binned_statistic_2d(
        x_valid, y_valid, z_valid,
        statistic='max',
        bins=[width, height]
    )

    # Rotate and flip to match DTM orientation
    chm_stat_rotated = np.flipud(np.fliplr(np.rot90(chm_stat, k=3)))
    
    # Replace NaN values with 0
    chm_stat_rotated = np.nan_to_num(chm_stat_rotated, nan=0)
    
    # Clip values to realistic range
    chm_stat_rotated = np.clip(chm_stat_rotated, 0, 50)
    
    return chm_stat_rotated

def calculate_lrm(dtm_array, sigma=5, mode='nearest', truncate=3.0, clip_range=(-2, 2)):
    # Create a mask for valid data
    valid_mask = ~np.isnan(dtm_array)
    
    # Fill NaN values with 0 for processing
    dtm_processed = np.nan_to_num(dtm_array, nan=0)
    
    # Apply Gaussian filter
    dtm_smoothed = gaussian_filter(dtm_processed, sigma=sigma, mode=mode, truncate=truncate)
    
    # Calculate LRM
    lrm = dtm_processed - dtm_smoothed
    
    #Clip values to specified range
    lrm = np.clip(lrm, clip_range[0], clip_range[1])    
    
    # Apply mask to preserve original data extent
    lrm[~valid_mask] = np.nan
    
    return lrm

def calculate_vdi(x, y, z, resolution, width, height):
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    mask = (z <= 12)
    z_sel = z[mask]
    x_sel = x[mask]
    y_sel = y[mask]

    if x_sel.size == 0 or y_sel.size == 0 or z_sel.size == 0:
        return np.zeros((height, width))

    z_low = z_sel[z_sel <= 0.8]
    x_low = x_sel[z_sel <= 0.8]
    y_low = y_sel[z_sel <= 0.8]

    total_density, _, _, _ = binned_statistic_2d(
        x_sel, y_sel, None, statistic='count', bins=[width, height]
    )
    low_density, _, _, _ = binned_statistic_2d(
        x_low, y_low, None, statistic='count', bins=[width, height]
    )

    vdi = np.divide(
        low_density, total_density, out=np.zeros_like(low_density), where=total_density > 0
    )

    vdi_rotated = np.flipud(np.fliplr(np.rot90(vdi, k=3)))
    vdi_interpolated = interpolate_missing_values(vdi_rotated, method='nearest')
    return vdi_interpolated

def interpolate_missing_values(grid, method='nearest'):
    x, y = np.indices(grid.shape)
    valid_mask = ~np.isnan(grid)
    points = np.array([x[valid_mask], y[valid_mask]]).T
    values = grid[valid_mask]

    interpolated_grid = griddata(
        points, values, (x, y), method=method, fill_value=0
    )
    return interpolated_grid

def normalize_percentile(data, low=1, high=99, nodata_value=0):
    datamask = data != nodata_value
    pmin = np.array([np.percentile(data[:, :, i][datamask[:, :, i]], q=low) for i in range(data.shape[-1])])
    pmax = np.array([np.percentile(data[:, :, i][datamask[:, :, i]], q=high) for i in range(data.shape[-1])])
    normalized_data = np.clip((data - pmin) / (pmax - pmin + 1E-10), 0, 1)
    normalized_data[~datamask] = nodata_value
    return normalized_data

def create_single_raster(data_array, transform, output_path, crs, nodata_value=0):
    height, width = data_array.shape
    with rasterio.open(
        output_path, "w",
        driver="GTiff", height=height, width=width,
        count=1, dtype=data_array.dtype, crs=crs, transform=transform,
        nodata=nodata_value,
        tiled=True,
        compress='lzw'
    ) as dst:
        dst.write(data_array, 1)
        # Ensure CRS is written to the file
        dst.crs = crs

def create_multiband_raster(data_arrays, transform, output_path, crs, nodata_value=0):
    num_bands = len(data_arrays)
    height, width = data_arrays[0].shape
    with rasterio.open(
        output_path, "w",
        driver="GTiff", height=height, width=width,
        count=num_bands, dtype=data_arrays[0].dtype, crs=crs, transform=transform,
        nodata=nodata_value,
        tiled=True,
        compress='lzw'
    ) as dst:
        for i, data_array in enumerate(data_arrays, start=1):
            dst.write(data_array, i)
        # Ensure CRS is written to the file
        dst.crs = crs

# Export the algorithm class
__all__ = ['TrailscanPreProcessingAlgorithm']