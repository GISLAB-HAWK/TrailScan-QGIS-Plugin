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

from typing import Any, Optional

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingParameterExpression,
    QgsProcessingParameterPointCloudLayer,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    Qgis,
    QgsRasterLayer,  
    QgsProcessingMultiStepFeedback,  
)
from qgis import processing
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d
from scipy.interpolate import griddata
import laspy
import rasterio
from rasterio.transform import rowcol
from rasterio.transform import from_origin
from rasterio.crs import CRS
import itertools

PIXEL_SIZE = 0.38  # Example pixel size, adjust as needed


class TrailscanInferenceProcessingAlgorithm(QgsProcessingAlgorithm):
    """

    """


    INPUT = "INPUT"
    OUTPUT = "OUTPUT"

    def name(self) -> str:
        """
        Returns the algorithm name
        """
        return "inference"

    def displayName(self) -> str:
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return "Inference"

    def group(self) -> str:
        """
        Returns the name of the group this algorithm belongs to. This string
        should be localised.
        """
        return "Processing"

    def groupId(self) -> str:
        """
        Returns the unique ID of the group this algorithm belongs to.
        """
        return "processing"

    def shortHelpString(self) -> str:
        """
        Returns a localised short helper string for the algorithm. 
        """
        return "Trailscan segmentation"

    def initAlgorithm(self, config: Optional[dict[str, Any]] = None):
        """
        Here we define the inputs and output of the algorithm, along
        with some other properties.
        """

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                name=self.INPUT,
                description="Input preprocessed point cloud data",
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterDestination(
                name=self.OUTPUT, 
                description="Normalized"
                )
        )            

    def calculate_extent_and_transform(self, input_laz, resolution):

        
        las = laspy.read(input_laz)
        x_min, x_max = np.min(las.x), np.max(las.x)
        y_min, y_max = np.min(las.y), np.max(las.y)

        # Calculate number of pixels
        width = int(np.ceil((x_max - x_min) / resolution))
        height = int(np.ceil((y_max - y_min) / resolution))

        # Transform object
        transform = from_origin(x_min, y_max, resolution, resolution)

        return transform, width, height

    def normalize_height(self, x, y, z, dtm_array, transform):
        
        rows, cols = rowcol(transform, x, y)
        z_normalized = z - dtm_array[rows, cols]
        return z_normalized

    def calculate_vdi(self, x, y, z, resolution, width, height):
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
        vdi_interpolated = self.interpolate_missing_values(vdi_rotated, method='nearest')
        return vdi_interpolated

    def interpolate_missing_values(self, grid, method='nearest'):
        x, y = np.indices(grid.shape)
        valid_mask = ~np.isnan(grid)
        points = np.array([x[valid_mask], y[valid_mask]]).T
        values = grid[valid_mask]

        interpolated_grid = griddata(
            points, values, (x, y), method=method, fill_value=0
        )
        return interpolated_grid 

    def create_single_raster(self, data_array, transform, output_path, crs, nodata_value=0):
        """Create a single-band raster from a data array.
        
        Args:
            data_array: Input data array
            output_path: Output file path
            crs: Coordinate reference system
            nodata_value: NoData value to use
        """
        height, width = data_array.shape
        crs_def = CRS.from_wkt(crs)

        with rasterio.open(
            output_path, "w",
            driver="GTiff", height=height, width=width,
            count=1, dtype=data_array.dtype, crs=crs_def, transform=transform,
            nodata=nodata_value
        ) as dst:
            dst.write(data_array, 1)    

    def normalize_percentile(self, data, low=1, high=99, nodata_value=0):
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

    def create_multiband_raster(self, data_arrays, transform, output_path, crs, nodata_value=0):
        """Create a multi-band raster from multiple data arrays.
        
        Args:
            data_arrays: List of input data arrays
            output_path: Output file path
            crs: Coordinate reference system
            nodata_value: NoData value to use
        """
        num_bands = len(data_arrays)
        height, width = data_arrays[0].shape
        crs_def = CRS.from_wkt(crs)

        with rasterio.open(
            output_path, "w",
            driver="GTiff", height=height, width=width,
            count=num_bands, dtype=data_arrays[0].dtype, crs=crs_def, transform=transform,
            nodata=nodata_value
        ) as dst:
            for i, data_array in enumerate(data_arrays, start=1):
                dst.write(data_array, i)                       

    def processAlgorithm(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingMultiStepFeedback,
    ) -> dict[str, Any]:
        """
        Here is where the processing itself takes place.
        
        """

        counter = itertools.count(1)
        count_max = 12
        feedback = QgsProcessingMultiStepFeedback(count_max, feedback)

        source = self.parameterAsRasterLayer(parameters, self.INPUT, context)

        output_raster = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)

        if source is None:
            raise QgsProcessingException(
                self.invalidSourceError(parameters, self.INPUT)
            )

        crs = source.crs().horizontalCrs()
        if not crs.isValid():
            raise QgsProcessingException("Invalid CRS in input raster")

        feedback.pushInfo(f"Using CRS: {crs.description()}")

        raster_out = {'OUTPUT': output_raster}

        

        return {self.OUTPUT: output_raster}

    def createInstance(self):
        return self.__class__()