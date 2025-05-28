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


# TODO: Installation von pip install laspy[lazrs,laszip] notwendig für LAS Backend

from typing import Any, Optional
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d
from scipy.interpolate import griddata
import laspy
import rasterio
from rasterio.transform import rowcol
from rasterio.transform import from_origin

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterPointCloudLayer,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsProcessingUtils,
    QgsRasterLayer
)
from qgis import processing
from qgis.core import QgsRasterFileWriter, QgsRasterDataProvider, QgsRasterBlock, QgsRaster
from osgeo import gdal, osr


class ExampleProcessingAlgorithm(QgsProcessingAlgorithm):
    """
    This is an example algorithm that takes a vector layer and
    creates a new identical one.

    It is meant to be used as an example of how to create your own
    algorithms and explain methods and variables used to do it. An
    algorithm like this will be available in all elements, and there
    is not need for additional work.

    All Processing algorithms should extend the QgsProcessingAlgorithm
    class.
    """

    # Constants used to refer to parameters and outputs. They will be
    # used when calling the algorithm from another algorithm, or when
    # calling from the QGIS console.

    INPUT = "INPUT"
    DTM = "DTM"
    OUTPUT_VDI = "OUTPUT_VDI"

    def name(self) -> str:
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return "myscript"

    def displayName(self) -> str:
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return "My Script"

    def group(self) -> str:
        """
        Returns the name of the group this algorithm belongs to. This string
        should be localised.
        """
        return "Example scripts"

    def groupId(self) -> str:
        """
        Returns the unique ID of the group this algorithm belongs to. This
        string should be fixed for the algorithm, and must not be localised.
        The group id should be unique within each provider. Group id should
        contain lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return "examplescripts"

    def shortHelpString(self) -> str:
        """
        Returns a localised short helper string for the algorithm. This string
        should provide a basic description about what the algorithm does and the
        parameters and outputs associated with it.
        """
        return "Example algorithm short description"

    def initAlgorithm(self, config: Optional[dict[str, Any]] = None):
        """
        Here we define the inputs and output of the algorithm, along
        with some other properties.
        """

        # We add the input vector features source. It can have any kind of
        # geometry.
        self.addParameter(
            QgsProcessingParameterPointCloudLayer(
                name=self.INPUT,
                description="Point cloud"
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                name=self.DTM,
                description="DTM"
            )
        )

        # We add a feature sink in which to store our processed features (this
        # usually takes the form of a newly created vector layer when the
        # algorithm is run in QGIS).
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                name=self.OUTPUT_VDI, 
                description="VDI"
                )
        )

    def calculate_extent_and_transform(self, input_laz, resolution):

        print(input_laz)
        
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

    # TODO: Coordinatensystem handling
    def create_single_raster(self, data_array, transform, output_path, crs="EPSG:25832", nodata_value=0):
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

    def processAlgorithm(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict[str, Any]:
        """
        Here is where the processing itself takes place.
        """

        # Retrieve the feature source and sink. The 'dest_id' variable is used
        # to uniquely identify the feature sink, and must be included in the
        # dictionary returned by the processAlgorithm function.
        source = self.parameterAsPointCloudLayer(parameters, self.INPUT, context)
        dtm = self.parameterAsRasterLayer(parameters, self.DTM, context)
        vdi = self.parameterAsOutputLayer(parameters, self.OUTPUT_VDI, context)

        # If source was not found, throw an exception to indicate that the algorithm
        # encountered a fatal error. The exception text can be any string, but in this
        # case we use the pre-built invalidSourceError method to return a standard
        # helper text for when a source cannot be evaluated
        if source is None:
            raise QgsProcessingException(
                self.invalidSourceError(parameters, self.INPUT)
            )

        dtm = dtm.dataProvider().dataSourceUri()
        input_laz = source.dataProvider().dataSourceUri()
        resolution = 0.38 

        feedback.pushInfo(f"Input LAZ file: {input_laz}")

        with rasterio.open(dtm) as src:
            dtm_array = src.read(1)
            nodata_value = src.nodata

        transform, width, height = self.calculate_extent_and_transform(input_laz, resolution)

        
        las = laspy.read(input_laz)

        x, y, z = las.x, las.y, las.z
        z_normalized = self.normalize_height(x, y, z, dtm_array, transform)

        vdi_array = self.calculate_vdi(x, y, z_normalized, resolution, width, height)
        self.create_single_raster(vdi_array, transform, vdi, nodata_value=nodata_value)


        # Register the output LRM raster
        vdi_out = {'OUTPUT': vdi}

        return {self.OUTPUT_VDI: vdi_out["OUTPUT"]}


    def createInstance(self):
        return self.__class__()
