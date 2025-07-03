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
import numpy as np
from scipy.ndimage import gaussian_filter

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterPointCloudLayer,
    QgsProcessingParameterRasterDestination,
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
    OUTPUT_DTM = "OUTPUT_DTM"
    OUTPUT_LRM = "OUTPUT_LRM"

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
                description="Input layer"
            )
        )

        # We add a feature sink in which to store our processed features (this
        # usually takes the form of a newly created vector layer when the
        # algorithm is run in QGIS).
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                name=self.OUTPUT_DTM, 
                description="DTM"
                )
        )

        self.addParameter(
            QgsProcessingParameterRasterDestination(
                name=self.OUTPUT_LRM,
                description="LRM",
            )
        )

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

        # If source was not found, throw an exception to indicate that the algorithm
        # encountered a fatal error. The exception text can be any string, but in this
        # case we use the pre-built invalidSourceError method to return a standard
        # helper text for when a source cannot be evaluated
        if source is None:
            raise QgsProcessingException(
                self.invalidSourceError(parameters, self.INPUT)
            )


        dtm = processing.run(
            "pdal:exportrastertin", 
                {
                    'INPUT': source,
                    'RESOLUTION':0.38,
                    'TILE_SIZE':1000,
                    'FILTER_EXPRESSION':'Classification = 2',
                    'FILTER_EXTENT':None,
                    'ORIGIN_X':None,
                    'ORIGIN_Y':None,
                    'OUTPUT':parameters['OUTPUT_DTM']},
            context=context,
            feedback=feedback,
            is_child_algorithm=True,
        )

        dtm_layer = QgsRasterLayer(dtm["OUTPUT"])
        dtm_array = dtm_layer.as_numpy(use_masking=False, bands=[0])
        dtm_array = dtm_array.reshape(-1, dtm_array.shape[1])


        dtm_smoothed_array = gaussian_filter(dtm_array, sigma=5, mode='reflect', truncate=3.0)
        lrm = dtm_array - dtm_smoothed_array



        # Prepare output path for LRM
        lrm_output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT_LRM, context)
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(lrm_output_path, lrm.shape[1], lrm.shape[0], 1, gdal.GDT_Float32)
    

        # Get pixel size from DTM layer using GDAL
        ds = gdal.Open(dtm["OUTPUT"])
        out_ds.SetGeoTransform(ds.GetGeoTransform())
        out_ds.SetProjection(ds.GetProjection())
        out_ds.GetRasterBand(1).WriteArray(lrm)

        
        out_ds.FlushCache()
        out_ds = None  # Close file

        # Register the output LRM raster
        lrm = {'OUTPUT': lrm_output_path}

        return {self.OUTPUT_DTM: dtm["OUTPUT"], self.OUTPUT_LRM: lrm["OUTPUT"]}


    def createInstance(self):
        return self.__class__()
