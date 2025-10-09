from typing import Any, Optional
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingParameterPointCloudLayer,
    QgsProcessingParameterRasterDestination,
    QgsRasterLayer,
    QgsProcessingMultiStepFeedback,
    QgsProcessingUtils,
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

PIXEL_SIZE = 0.25
DTM_PIPELINE = "dtm_pipeline.json"
CHM_PIPELINE = "chm_pipeline.json"
LOW_VEGETATION_PIPELINE = "low_vegetation_pipeline.json"
HIGH_VEGETATION_PIPELINE = "high_vegetation_pipeline.json"


class TrailscanPreProcessingAlgorithm(QgsProcessingAlgorithm):

    POINTCLOUD = "POINTCLOUD"
    OUTPUT_NORMALIZED = "OUTPUT_NORMALIZED"

    def name(self) -> str:
        return "preprocessing"

    def displayName(self) -> str:
        return "01 Preprocessing Point Cloud"

    def group(self) -> str:
        return ""

    def groupId(self) -> str:
        return ""

    def shortHelpString(self) -> str:
        return (
            "The ALS point cloud (.laz or .las format) is processed with the TrailScan preprocessing tool.\n\n"
            "The point cloud is converted into a 4-band georeferenced raster image:\n"
            "- Band 1: Digital Terrain Model (DTM)\n"
            "- Band 2: Canopy Height Model (CHM)\n"
            "- Band 3: Micro Relief Model (MRM)\n"
            "- Band 4: Vegetation Density Index (VDI)\n\n"
            "Each band's values are normalized to the range 0–1.\n"
            "The output file is therefore named 'Normalized'."
        )

    def icon(self):
        return QIcon(os.path.join(os.path.dirname(__file__), 'TrailScan_Logo.svg'))

    def initAlgorithm(self, config: Optional[dict[str, Any]] = None):
        # Input Point Cloud
        self.addParameter(
            QgsProcessingParameterPointCloudLayer(
                name=self.POINTCLOUD,
                description="Input point cloud"
            )
        )

        # Main 4-Band Normalized Raster Output
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                name=self.OUTPUT_NORMALIZED,
                description="Normalized"
            )
        )

    # helper functions
    def calculate_extent_and_transform(self, input_laz, resolution):
        las = laspy.read(input_laz)
        x_min, x_max = np.min(las.x), np.max(las.x)
        y_min, y_max = np.min(las.y), np.max(las.y)
        width = int(np.ceil((x_max - x_min) / resolution))
        height = int(np.ceil((y_max - y_min) / resolution))
        transform = from_origin(x_min, y_max, resolution, resolution)
        return transform, width, height

    def create_single_raster(self, data_array, transform, output_path, crs, nodata_value=0):
        height, width = data_array.shape
        crs_def = CRS.from_wkt(crs)
        with rasterio.open(
            output_path, "w",
            driver="GTiff", height=height, width=width,
            count=1, dtype=data_array.dtype, crs=crs_def, transform=transform,
            nodata=nodata_value
        ) as dst:
            dst.write(data_array, 1)

    def create_multiband_raster(self, data_arrays, transform, output_path, crs, nodata_value=0):
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
            dst.descriptions = ["DTM", "CHM", "MRM", "VDI"]

    def normalize_percentile(self, data, low=1, high=99, nodata_value=0):
        datamask = data != nodata_value
        pmin = np.array([np.percentile(data[:, :, i][datamask[:, :, i]], q=low) for i in range(data.shape[-1])])
        pmax = np.array([np.percentile(data[:, :, i][datamask[:, :, i]], q=high) for i in range(data.shape[-1])])
        normalized_data = np.clip((data - pmin) / (pmax - pmin + 1E-10), 0, 1)
        normalized_data[~datamask] = nodata_value
        return normalized_data

    def processAlgorithm(self, parameters, context, feedback):

        counter = itertools.count(1)
        count_max = 8
        feedback = QgsProcessingMultiStepFeedback(count_max, feedback)

        sourceCloud = self.parameterAsPointCloudLayer(parameters, self.POINTCLOUD, context)
        input_laz = sourceCloud.dataProvider().dataSourceUri()
        if isinstance(input_laz, str) and input_laz.lower().startswith("pdal://"):
            input_laz = input_laz[len("pdal://"):]

        if sourceCloud is None:
            raise QgsProcessingException("Invalid point cloud input")

        # --- Load LAS ---
        las = laspy.read(input_laz)

        # --- Initialize errors/warnings ---
        errors = []
        warnings = []

        # --- Check point density ---
        x_min, x_max = np.min(las.x), np.max(las.x)
        y_min, y_max = np.min(las.y), np.max(las.y)
        area_m2 = (x_max - x_min) * (y_max - y_min)
        point_density = len(las.x) / area_m2
        feedback.pushInfo(f"Point density: {point_density:.2f} pts/m²")

        if point_density < 4:
            errors.append("Point density too low (<4 pts/m²). Please use a higher resolution dataset.")
        elif point_density > 20:
            warnings.append(
                "High point density (>20 pts/m²). Processing may be slow. "
                "Consider thinning the point cloud, e.g. using the QGIS tool 'Thin'."
            )

        # --- Check classification ---
        if not hasattr(las, "classification"):
            errors.append(
                "Point cloud is not classified. Ground points must be class 2; others must have distinct classes."
            )
        else:
            try:
                classes = np.unique(las.classification).astype(int)
            except Exception:
                classes = np.unique(np.array(las.classification, dtype=int))

            feedback.pushInfo(f"Detected classification codes: {classes.tolist()}")

            if 2 not in classes:
                errors.append("No ground points found (class 2 missing). Ground points must be class 2.")
            if not np.any(classes != 2):
                errors.append(
                    "Point cloud contains only ground points (only class 2). Need at least one additional class (e.g., vegetation).")

        # --- Check CRS ---
        crs = sourceCloud.crs().horizontalCrs()
        if not crs.isValid():
            project_crs = context.project().crs() if context.project() else None
            if project_crs and project_crs.isValid():
                crs = project_crs
                warnings.append("No valid CRS found in point cloud. Using project CRS.")
            else:
                errors.append("Missing CRS. Please specify a valid CRS.")

        # --- Output warnings and errors ---
        for w in warnings:
            feedback.pushWarning(w)

        if errors:
            raise QgsProcessingException("\n".join(errors))

        # --- Check PDAL availability ---
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            subprocess.run(["pdal", "--version"], check=True, creationflags=creationflags)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise QgsProcessingException("PDAL not installed or not found in PATH.")

        # --- Temporary output files ---
        dtm_outfile = QgsProcessingUtils.generateTempFilename("DTM.tif", context=context)
        chm_outfile = QgsProcessingUtils.generateTempFilename("CHM.tif", context=context)
        mrm_outfile = QgsProcessingUtils.generateTempFilename("MRM.tif", context=context)
        vdi_outfile = QgsProcessingUtils.generateTempFilename("VDI.tif", context=context)
        low_vegetation_outfile = QgsProcessingUtils.generateTempFilename("LowVegetation.tif", context=context)
        high_vegetation_outfile = QgsProcessingUtils.generateTempFilename("HighVegetation.tif", context=context)
        output_raster = self.parameterAsOutputLayer(parameters, self.OUTPUT_NORMALIZED, context)

        # --- Run DTM pipeline ---
        feedback.pushInfo("Running DTM pipeline...")
        dtm_pipeline_path = os.path.join(os.path.dirname(__file__), DTM_PIPELINE)
        subprocess.run([
            "pdal", "pipeline", dtm_pipeline_path,
            f"--readers.las.filename={input_laz}",
            f"--writers.gdal.filename={dtm_outfile}",
            f"--writers.gdal.resolution={PIXEL_SIZE}"
        ], check=True, creationflags=creationflags)

        feedback.setCurrentStep(next(counter))
        if feedback.isCanceled():
            return {}

        # --- Run CHM pipeline ---
        feedback.pushInfo("Running CHM pipeline...")
        chm_pipeline_path = os.path.join(os.path.dirname(__file__), CHM_PIPELINE)
        subprocess.run([
            "pdal", "pipeline", chm_pipeline_path,
            f"--readers.las.filename={input_laz}",
            f"--writers.gdal.filename={chm_outfile}",
            f"--writers.gdal.resolution={PIXEL_SIZE}"
        ], check=True, creationflags=creationflags)

        feedback.setCurrentStep(next(counter))
        if feedback.isCanceled():
            return {}

        # --- Load rasters ---
        with rasterio.open(dtm_outfile) as dtm_src:
            dtm_array = dtm_src.read(1)
            nodata_value = dtm_src.nodata or 0
            transform = dtm_src.transform
        with rasterio.open(chm_outfile) as chm_src:
            chm_array = chm_src.read(1)

        # --- Align arrays to same shape ---
        min_rows = min(dtm_array.shape[0], chm_array.shape[0])
        min_cols = min(dtm_array.shape[1], chm_array.shape[1])

        dtm_array = dtm_array[:min_rows, :min_cols]
        chm_array = chm_array[:min_rows, :min_cols]

        # MRM based on DTM
        dtm_smoothed_array = median_filter(dtm_array, size=10)
        mrm_array = np.clip(dtm_array - dtm_smoothed_array, -1, 1)
        mrm_array = mrm_array[:min_rows, :min_cols]

        feedback.setCurrentStep(next(counter))
        if feedback.isCanceled():
            return {}

        # --- Calculate low vegetation ---
        feedback.pushInfo("Calculating low vegetation...")
        low_pipeline_path = os.path.join(os.path.dirname(__file__), LOW_VEGETATION_PIPELINE)
        subprocess.run([
            "pdal", "pipeline", low_pipeline_path,
            f"--readers.las.filename={input_laz}",
            f"--writers.gdal.filename={low_vegetation_outfile}",
            f"--writers.gdal.resolution={PIXEL_SIZE}"
        ], check=True, creationflags=creationflags)

        feedback.setCurrentStep(next(counter))
        if feedback.isCanceled():
            return {}

        # --- Calculate high vegetation ---
        feedback.pushInfo("Calculating high vegetation...")
        high_pipeline_path = os.path.join(os.path.dirname(__file__), HIGH_VEGETATION_PIPELINE)
        subprocess.run([
            "pdal", "pipeline", high_pipeline_path,
            f"--readers.las.filename={input_laz}",
            f"--writers.gdal.filename={high_vegetation_outfile}",
            f"--writers.gdal.resolution={PIXEL_SIZE}"
        ], check=True, creationflags=creationflags)

        feedback.setCurrentStep(next(counter))
        if feedback.isCanceled():
            return {}

        # --- Calculate VDI ---
        feedback.pushInfo("Calculating VDI...")
        with rasterio.open(low_vegetation_outfile) as low_veg_src:
            low_veg_array = low_veg_src.read(1)
        with rasterio.open(high_vegetation_outfile) as high_veg_src:
            high_veg_array = high_veg_src.read(1)

        # --- Align shapes (avoid broadcasting errors) ---
        min_rows = min(low_veg_array.shape[0], high_veg_array.shape[0])
        min_cols = min(low_veg_array.shape[1], high_veg_array.shape[1])
        low_veg_array = low_veg_array[:min_rows, :min_cols]
        high_veg_array = high_veg_array[:min_rows, :min_cols]

        # --- Compute VDI (ratio low/high vegetation) ---
        with np.errstate(divide='ignore', invalid='ignore'):
            vdi_array = np.divide(
                low_veg_array.astype(np.float32),
                high_veg_array.astype(np.float32),
                out=np.zeros((min_rows, min_cols), dtype=np.float32),
                where=high_veg_array != 0,
            )

        # Replace zeros with small constant to avoid log issues later
        vdi_array = np.where(vdi_array == 0, 0.1, vdi_array)
        vdi_array = np.clip(vdi_array, 0, 1)


        feedback.setCurrentStep(next(counter))
        if feedback.isCanceled():
            return {}

        # --- Create normalized multiband raster ---
        feedback.pushInfo("Creating normalized multiband raster...")
        combined_array = np.stack([dtm_array, chm_array, mrm_array, vdi_array], axis=2)
        normalized_array = self.normalize_percentile(combined_array, nodata_value=nodata_value)
        self.create_multiband_raster(
            [normalized_array[:, :, i] for i in range(4)],
            transform,
            output_raster,
            crs.toWkt(),
            nodata_value
        )

        feedback.setCurrentStep(count_max)
        return {self.OUTPUT_NORMALIZED: output_raster}

    def createInstance(self):
        return self.__class__()
