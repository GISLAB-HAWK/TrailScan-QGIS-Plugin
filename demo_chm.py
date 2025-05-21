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
    QgsProcessingFeedback,
    QgsProcessingParameterPointCloudLayer,
    QgsProcessingParameterPointCloudAttribute,
    QgsProcessingParameterRasterDestination,
    QgsProcessingUtils,
    QgsRasterLayer
)
from qgis import processing


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
    POINT_CLASSES = "POINT_CLASSES"
    OUTPUT = "OUTPUT"

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

        # self.addParameter(
        #     QgsProcessingParameterPointCloudAttribute(
        #         name=self.POINT_CLASSES,
        #         description="Point classes",
        #         parentLayerParameterName=self.INPUT,
        #     )

        # )

        self.addParameter(
            QgsProcessingParameterRasterDestination(self.OUTPUT, "DSM")
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


        dsm = processing.run(
            "pdal:exportrastertin", 
                {
                    'INPUT': source,
                    'RESOLUTION':0.38,
                    'TILE_SIZE':1000,
                    'FILTER_EXPRESSION':'Classification IN (3, 4, 5)',
                    'FILTER_EXTENT':None,
                    'ORIGIN_X':None,
                    'ORIGIN_Y':None,
                    'OUTPUT':'TEMPORARY_OUTPUT'},
            context=context,
            feedback=feedback,
            is_child_algorithm=True,
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
                    'OUTPUT':'TEMPORARY_OUTPUT'},
            context=context,
            feedback=feedback,
            is_child_algorithm=True,
        )
        
        dsm_layer = QgsRasterLayer(dsm["OUTPUT"])
        dsm_layer.setName("dsm")
        dtm_layer = QgsRasterLayer(dtm["OUTPUT"])
        dtm_layer.setName("dtm")

        chm = processing.run(
            "native:rastercalc", 
                {
                'LAYERS':[dsm_layer, dtm_layer],
                'EXPRESSION': '"dsm@1"-"dtm@1"',
                'EXTENT':None,
                'CELL_SIZE':None,
                'CRS':None,
                'OUTPUT':parameters['OUTPUT']},
            context=context,
            feedback=feedback,
            is_child_algorithm=True,
        )

        return {self.OUTPUT: chm["OUTPUT"]}


    def createInstance(self):
        return self.__class__()
