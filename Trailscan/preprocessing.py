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
    Qgis,
    QgsRasterLayer,  
    QgsProcessingMultiStepFeedback,  
)
from qgis.PyQt.QtGui import QIcon
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
import os

PIXEL_SIZE = 0.38  # Example pixel size, adjust as needed


class TrailscanPreProcessingAlgorithm(QgsProcessingAlgorithm):
    """

    """


    INPUT = "INPUT"
    OUTPUT = "OUTPUT"
    POINTCLOUD = "POINTCLOUD"
    EXPRESSION_DSM = "EXPRESSION_DSM"
    EXPRESSION_DTM = "EXPRESSION_DTM"
    OUTPUT_DTM = "OUTPUT_DTM"
    OUTPUT_DSM = "OUTPUT_DSM"
    OUTPUT_LRM = "OUTPUT_LRM"
    OUTPUT_CHM = "OUTPUT_CHM"
    OUTPUT_VDI = "OUTPUT_VDI"
    OUTPUT_NORMALIZED = "OUTPUT_NORMALIZED"

    def name(self) -> str:
        """
        Returns the algorithm name
        """
        return "preprocessing"

    def displayName(self) -> str:
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return "01 Preprocessing Point Cloud"

    def group(self) -> str:
        """
        Returns the name of the group this algorithm belongs to. This string
        should be localised.
        """
        return ""

    def groupId(self) -> str:
        """
        Returns the unique ID of the group this algorithm belongs to.
        """
        return ""

    def shortHelpString(self) -> str:
        """
        Returns a localised short helper string for the algorithm. 
        """
        return "Trailscan preprocessing algorithm"

    def icon(self):
        return QIcon(os.path.join(os.path.dirname(__file__), 'TrailScan_Logo.svg'))

    def initAlgorithm(self, config: Optional[dict[str, Any]] = None):
        """
        Here we define the inputs and output of the algorithm, along
        with some other properties.
        """

        self.addParameter(
            QgsProcessingParameterPointCloudLayer(
                name=self.POINTCLOUD,
                description="Input point cloud"
            )
        )

        self.addParameter(
            QgsProcessingParameterExpression(
                name=self.EXPRESSION_DTM,
                description="Selection DTM classes",
                parentLayerParameterName=self.POINTCLOUD,
                defaultValue="Classification IN (2)",
                optional=False,
                type=Qgis.ExpressionType.PointCloud
            )
        )  

        self.addParameter(
            QgsProcessingParameterExpression(
                name=self.EXPRESSION_DSM,
                description="Selection DSM classes",
                parentLayerParameterName=self.POINTCLOUD,
                defaultValue="Classification IN (2, 20)",
                optional=False,
                type=Qgis.ExpressionType.PointCloud
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterDestination(
                name=self.OUTPUT_DTM, 
                description="DTM"
                )
        )

        self.addParameter(
            QgsProcessingParameterRasterDestination(
                name=self.OUTPUT_DSM, 
                description="DSM"
                )
        ) 

        self.addParameter(
            QgsProcessingParameterRasterDestination(
                name=self.OUTPUT_LRM,
                description="LRM",
            )
        )  

        self.addParameter(
            QgsProcessingParameterRasterDestination(
                name=self.OUTPUT_CHM,
                description="CHM",
            )
        )         

        self.addParameter(
            QgsProcessingParameterRasterDestination(
                name=self.OUTPUT_VDI, 
                description="VDI"
                )
        ) 

        self.addParameter(
            QgsProcessingParameterRasterDestination(
                name=self.OUTPUT_NORMALIZED, 
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

        sourceCloud = self.parameterAsPointCloudLayer(parameters, self.POINTCLOUD, context)
        input_laz = sourceCloud.dataProvider().dataSourceUri()
        classification_dsm = self.parameterAsExpression(parameters, self.EXPRESSION_DSM, context)
        classification_dtm = self.parameterAsExpression(parameters, self.EXPRESSION_DTM, context)
        vdi_outfile = self.parameterAsOutputLayer(parameters, self.OUTPUT_VDI, context)
        dtm_outfile = self.parameterAsOutputLayer(parameters, self.OUTPUT_DTM, context) 
        dsm_outfile = self.parameterAsOutputLayer(parameters, self.OUTPUT_DSM, context)
        lrm_outfile = self.parameterAsOutputLayer(parameters, self.OUTPUT_LRM, context)
        chm_outfile = self.parameterAsOutputLayer(parameters, self.OUTPUT_CHM, context)
        output_raster = self.parameterAsOutputLayer(parameters, self.OUTPUT_NORMALIZED, context)

        if sourceCloud is None:
            raise QgsProcessingException(
                self.invalidSourceError(parameters, self.POINTCLOUD)
            )

        crs = sourceCloud.crs().horizontalCrs()
        if not crs.isValid():
            raise QgsProcessingException("Invalid CRS in input point cloud")
        
        feedback.pushInfo(f"Using CRS: {crs.description()}")

        dtm = processing.run(
            "pdal:exportrastertin", 
                {
                    'INPUT': sourceCloud,
                    'RESOLUTION':PIXEL_SIZE,
                    'TILE_SIZE':1000,
                    'FILTER_EXPRESSION': classification_dtm,
                    'FILTER_EXTENT':None,
                    'ORIGIN_X':None,
                    'ORIGIN_Y':None,
                    'OUTPUT':dtm_outfile},
            context=context,
            feedback=feedback,
            is_child_algorithm=True,
        )

        feedback.setCurrentStep(next(counter))
        if feedback.isCanceled():
            return {}

        dsm = processing.run(
            "pdal:exportrastertin", 
                {
                    'INPUT': sourceCloud,
                    'RESOLUTION':PIXEL_SIZE,
                    'TILE_SIZE':1000,
                    'FILTER_EXPRESSION': classification_dsm,
                    'FILTER_EXTENT':None,
                    'ORIGIN_X':None,
                    'ORIGIN_Y':None,
                    'OUTPUT':dsm_outfile},
            context=context,
            feedback=feedback,
            is_child_algorithm=True,
        )

        feedback.setCurrentStep(next(counter))
        if feedback.isCanceled():
            return {}

        dtm_layer = QgsRasterLayer(dtm["OUTPUT"])        
        dtm_in = dtm_layer.dataProvider().dataSourceUri()
        
        with rasterio.open(dtm_in) as src:
            dtm_array = src.read(1)
            nodata_value = src.nodata

        dtm_layer = QgsRasterLayer(dtm["OUTPUT"])
        dtm_array = dtm_layer.as_numpy(use_masking=False, bands=[0])
        dtm_array = dtm_array.reshape(-1, dtm_array.shape[1])

        dsm_array = QgsRasterLayer(dsm["OUTPUT"]).as_numpy(use_masking=False, bands=[0])
        dsm_array = dsm_array.reshape(-1, dsm_array.shape[1])

        # CHM calculation
        chm_array = dsm_array - dtm_array

        # Collect information for raster creation
        transform, width, height = self.calculate_extent_and_transform(input_laz, PIXEL_SIZE)
        width = dtm_layer.width() 
        height = dtm_layer.height()

        # Write CHM raster
        self.create_single_raster(chm_array, transform, chm_outfile, crs.toWkt(), nodata_value=nodata_value)

        feedback.setCurrentStep(next(counter))
        if feedback.isCanceled():
            return {}

        dtm_smoothed_array = gaussian_filter(dtm_array, sigma=5, mode='reflect', truncate=3.0)
        lrm_array = dtm_array - dtm_smoothed_array
        lrm_array = np.clip(lrm_array, -1, 1)

        self.create_single_raster(lrm_array, transform, lrm_outfile, crs.toWkt(), nodata_value=nodata_value)

        feedback.setCurrentStep(next(counter))
        if feedback.isCanceled():
            return {}
        
        las = laspy.read(input_laz)

        x, y, z = las.x, las.y, las.z
        z_normalized = self.normalize_height(x, y, z, dtm_array, transform)

        feedback.setCurrentStep(next(counter))
        if feedback.isCanceled():
            return {}

        vdi_array = self.calculate_vdi(x, y, z_normalized, PIXEL_SIZE, width, height)
        self.create_single_raster(vdi_array, transform, vdi_outfile, crs.toWkt(), nodata_value=nodata_value)

        feedback.setCurrentStep(next(counter))
        if feedback.isCanceled():
            return {}

        combined_array = np.stack([dtm_array, chm_array, lrm_array, vdi_array], axis=2)
        normalized_array = self.normalize_percentile(combined_array, nodata_value=nodata_value)

        self.create_multiband_raster(
            [normalized_array[:,:,i] for i in range(4)], 
            transform, 
            output_raster,
            crs.toWkt(),
            nodata_value=nodata_value
        )

        feedback.setCurrentStep(count_max)

        # Register the output raster
        dtm_out = {'OUTPUT': dtm_outfile}
        dsm_out = {'OUTPUT': dsm_outfile}
        vdi_out = {'OUTPUT': vdi_outfile}
        lrm_out = {'OUTPUT': lrm_outfile}
        chm_out = {'OUTPUT': chm_outfile}
        raster_out = {'OUTPUT': output_raster}

        

        return {self.OUTPUT_DTM: dtm_out["OUTPUT"], self.OUTPUT_LRM: lrm_out["OUTPUT"], 
        self.OUTPUT_CHM: chm_out["OUTPUT"], self.OUTPUT_DSM: dsm_out["OUTPUT"], 
        self.OUTPUT_VDI: vdi_out["OUTPUT"], self.OUTPUT_NORMALIZED: raster_out["OUTPUT"]}

    def createInstance(self):
        return self.__class__()
