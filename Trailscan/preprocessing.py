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
    QgsProcessingParameterPointCloudLayer,
    QgsProcessingParameterRasterDestination,
    QgsRasterLayer,  
    QgsProcessingMultiStepFeedback,  
)
from qgis.PyQt.QtGui import QIcon
from scipy.ndimage import median_filter
from rasterio.transform import from_origin
from rasterio.crs import CRS
import numpy as np
import laspy
import rasterio
import itertools
import os
import subprocess

PIXEL_SIZE = 0.38  # Example pixel size, adjust as needed
DTM_PIPELINE = "dtm_pipeline.json"
CHM_PIPELINE = "chm_pipeline.json"
LOW_VEGETATION_PIPELINE = "low_vegetation_pipeline.json"
HIGH_VEGETATION_PIPELINE = "high_vegetation_pipeline.json"


class TrailscanPreProcessingAlgorithm(QgsProcessingAlgorithm):
    """
    Preparation of point cloud data for Trailscan analysis.
    This algorithm processes point cloud data to create various raster outputs
    such as DTM, CHM, LRM and VDI.
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
    OUTPUT_HIGH_VEGETATION = "OUTPUT_HIGH_VEGETATION"
    OUTPUT_LOW_VEGETATION = "OUTPUT_LOW_VEGETATION"

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
                name=self.OUTPUT_HIGH_VEGETATION, 
                description="High Vegetation"
                )
        ) 

        self.addParameter(
            QgsProcessingParameterRasterDestination(
                name=self.OUTPUT_LOW_VEGETATION, 
                description="Low Vegetation"
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
        count_max = 10
        feedback = QgsProcessingMultiStepFeedback(count_max, feedback)

        sourceCloud = self.parameterAsPointCloudLayer(parameters, self.POINTCLOUD, context)
        input_laz = sourceCloud.dataProvider().dataSourceUri()
        vdi_outfile = self.parameterAsOutputLayer(parameters, self.OUTPUT_VDI, context)
        dtm_outfile = self.parameterAsOutputLayer(parameters, self.OUTPUT_DTM, context) 
        lrm_outfile = self.parameterAsOutputLayer(parameters, self.OUTPUT_LRM, context)
        chm_outfile = self.parameterAsOutputLayer(parameters, self.OUTPUT_CHM, context)
        low_vegetation_outfile = self.parameterAsOutputLayer(parameters, self.OUTPUT_LOW_VEGETATION, context)
        high_vegetation_outfile = self.parameterAsOutputLayer(parameters, self.OUTPUT_HIGH_VEGETATION, context)
        output_raster = self.parameterAsOutputLayer(parameters, self.OUTPUT_NORMALIZED, context)

        if sourceCloud is None:
            raise QgsProcessingException(
                self.invalidSourceError(parameters, self.POINTCLOUD)
            )

        crs = sourceCloud.crs().horizontalCrs()
        if not crs.isValid():
            raise QgsProcessingException("Invalid CRS in input point cloud")

        # Check if PDAL is installed
        try:
            subprocess.run(["pdal", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise QgsProcessingException("PDAL is not installed or not found in PATH. Please install PDAL to continue.")
        
        feedback.setCurrentStep(next(counter))
        if feedback.isCanceled():
            return {}

        feedback.pushInfo("Creating DTM...")

        # Ensure output directories exist
        for outfile in [dtm_outfile, lrm_outfile, chm_outfile, vdi_outfile, low_vegetation_outfile, high_vegetation_outfile, output_raster]:
            out_dir = os.path.dirname(outfile)
            if out_dir and not os.path.isdir(out_dir):
                os.makedirs(out_dir, exist_ok=True)

       # Define CREATE_NO_WINDOW only on Windows to suppress the console window
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)                

        # Run DTM PDAL pipeline
        dtm_pipeline_path = os.path.join(os.path.dirname(__file__), DTM_PIPELINE)
        
        try:
            subprocess.run(
                [
                    "pdal", "pipeline", dtm_pipeline_path,
                    f"--readers.las.filename={input_laz}",
                    f"--writers.gdal.filename={dtm_outfile}",
                    f"--writers.gdal.resolution={PIXEL_SIZE}",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=creationflags
            )
        except subprocess.CalledProcessError as e:
            feedback.reportError(e.stderr or str(e))
            raise QgsProcessingException("PDAL DTM pipeline failed. See log for details.")
        if not os.path.exists(dtm_outfile):
            raise QgsProcessingException(f"DTM output file was not created: {dtm_outfile}")

        feedback.setCurrentStep(next(counter))
        if feedback.isCanceled():
            return {}

        feedback.pushInfo("Creating CHM...")

        chm_pipeline_path = os.path.join(os.path.dirname(__file__), CHM_PIPELINE)
        try:
            subprocess.run(
                [
                    "pdal", "pipeline", chm_pipeline_path,
                    f"--readers.las.filename={input_laz}",
                    f"--writers.gdal.filename={chm_outfile}",
                    f"--writers.gdal.resolution={PIXEL_SIZE}",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=creationflags,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            feedback.reportError(e.stderr or str(e))
            raise QgsProcessingException("PDAL CHM pipeline failed. See log for details.")
        if not os.path.exists(chm_outfile):
            raise QgsProcessingException(f"CHM output file was not created: {chm_outfile}")

        # Load rasters
        with rasterio.open(chm_outfile) as chm_src:
            chm_array = chm_src.read(1)

        with rasterio.open(dtm_outfile) as dtm_src:
            dtm_array = dtm_src.read(1)
            nodata_value = dtm_src.nodata if dtm_src.nodata is not None else 0


        # Collect information for raster creation
        transform, width, height = self.calculate_extent_and_transform(input_laz, PIXEL_SIZE)

        feedback.setCurrentStep(next(counter))
        if feedback.isCanceled():
            return {}

        feedback.pushInfo("Calculating LRM...")
        dtm_smoothed_array = median_filter(dtm_array, size=10)
        lrm_array = dtm_array - dtm_smoothed_array
        lrm_array = np.clip(lrm_array, -1, 1)

        self.create_single_raster(lrm_array, transform, lrm_outfile, crs.toWkt(), nodata_value=nodata_value)

        feedback.setCurrentStep(next(counter))
        if feedback.isCanceled():
            return {}

        feedback.pushInfo("Calculating low and high vegetation...")

        try:
            subprocess.run(
                [
                    "pdal", "pipeline", os.path.join(os.path.dirname(__file__), LOW_VEGETATION_PIPELINE),
                    f"--readers.las.filename={input_laz}",
                    f"--writers.gdal.filename={low_vegetation_outfile}",
                    f"--writers.gdal.resolution={PIXEL_SIZE}",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=creationflags,
            )
        except subprocess.CalledProcessError as e:
            feedback.reportError(e.stderr or str(e))
            raise QgsProcessingException("PDAL low vegetation pipeline failed. See log for details.")
        if not os.path.exists(low_vegetation_outfile):
            raise QgsProcessingException(f"Low vegetation output file was not created: {low_vegetation_outfile}")

        feedback.setCurrentStep(next(counter))
        if feedback.isCanceled():
            return {}

        try:
            subprocess.run(
                [
                    "pdal", "pipeline", os.path.join(os.path.dirname(__file__), HIGH_VEGETATION_PIPELINE),
                    f"--readers.las.filename={input_laz}",
                    f"--writers.gdal.filename={high_vegetation_outfile}",
                    f"--writers.gdal.resolution={PIXEL_SIZE}",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=creationflags,
            )
        except subprocess.CalledProcessError as e:
            feedback.reportError(e.stderr or str(e))
            raise QgsProcessingException("PDAL high vegetation pipeline failed. See log for details.")
        if not os.path.exists(high_vegetation_outfile):
            raise QgsProcessingException(f"High vegetation output file was not created: {high_vegetation_outfile}")

        feedback.setCurrentStep(next(counter))
        if feedback.isCanceled():
            return {}

        feedback.pushInfo("Calculating VDI...")

        with rasterio.open(low_vegetation_outfile) as low_veg_src:
            low_veg_array = low_veg_src.read(1)
        with rasterio.open(high_vegetation_outfile) as high_veg_src:
            high_veg_array = high_veg_src.read(1)   

        with np.errstate(divide='ignore', invalid='ignore'):
            vdi_array = np.divide(
                low_veg_array.astype(np.float32),
                high_veg_array.astype(np.float32),
                out=np.zeros_like(low_veg_array, dtype=np.float32),
                where=high_veg_array != 0,
            )
        self.create_single_raster(vdi_array, transform, vdi_outfile, crs.toWkt(), nodata_value=nodata_value)

        feedback.setCurrentStep(next(counter))
        if feedback.isCanceled():
            return {}

        feedback.pushInfo("Creating normalized raster by combining the results...")

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
        vdi_out = {'OUTPUT': vdi_outfile}
        lrm_out = {'OUTPUT': lrm_outfile}
        chm_out = {'OUTPUT': chm_outfile}
        raster_out = {'OUTPUT': output_raster}
        high_veg_out = {'OUTPUT': high_vegetation_outfile}
        low_veg_out = {'OUTPUT': low_vegetation_outfile}    

        return {self.OUTPUT_DTM: dtm_out["OUTPUT"], self.OUTPUT_LRM: lrm_out["OUTPUT"], 
        self.OUTPUT_CHM: chm_out["OUTPUT"], 
        self.OUTPUT_VDI: vdi_out["OUTPUT"], self.OUTPUT_NORMALIZED: raster_out["OUTPUT"],
        self.OUTPUT_HIGH_VEGETATION: high_veg_out["OUTPUT"], self.OUTPUT_LOW_VEGETATION: low_veg_out["OUTPUT"]}

    def createInstance(self):
        return self.__class__()
