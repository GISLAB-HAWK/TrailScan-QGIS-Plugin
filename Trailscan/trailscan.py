#-----------------------------------------------------------
# Copyright (C) 2025 HAWK Forest Monitoring Lab (Tanja Kempen, Mathias Groebe)
#-----------------------------------------------------------
# Licensed under the terms of GNU GPL 2
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#---------------------------------------------------------------------

import os
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QAction
from qgis.core import QgsApplication, QgsProcessingProvider, QgsProcessingAlgorithm, Qgis
import processing

def classFactory(iface):
    """Load TrailScan plugin."""
    return TrailScan(iface)

# --------------------------------------------------------------------
class TrailScan:
    """Main plugin class for TrailScan."""

    def __init__(self, iface):
        self.iface = iface
        self.algorithms_loaded = False

    def initGui(self):
        """Initialize toolbar and actions."""
        # Create toolbar
        self.toolbar = self.iface.addToolBar("TrailScan")
        self.toolbar.setObjectName("TrailScanToolbar")

        # Define toolbar icons
        icon_preprocessing = os.path.join(os.path.dirname(__file__), 'TrailScan_Logo.png')
        icon_inference = os.path.join(os.path.dirname(__file__), 'TrailScan_Inference.png')

        # Preprocessing action
        self.action = QAction(QIcon(icon_preprocessing), 'Preprocessing', self.iface.mainWindow())
        self.action.setIconText('Preprocessing')
        self.action.setToolTip('Run TrailScan Preprocessing')
        self.action.setStatusTip('Preprocess ALS point clouds for TrailScan')

        # Inference action
        self.action2 = QAction(QIcon(icon_inference), 'Inference', self.iface.mainWindow())
        self.action2.setIconText('Inference')
        self.action2.setToolTip('Run TrailScan Inference')
        self.action2.setStatusTip('Run TrailScan model inference on preprocessed raster')

        # Show text next to icon
        self.toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        # Add actions to toolbar
        self.toolbar.addAction(self.action)
        self.toolbar.addAction(self.action2)

        # Connect actions
        self.action.triggered.connect(self.runPreProcessing)
        self.action2.triggered.connect(self.runInference)

        # Initialize Processing Provider
        self.initProcessing()

    def initProcessing(self):
        """Register TrailScan Processing Provider."""
        try:
            self.provider = TrailScanProvider()
            QgsApplication.processingRegistry().addProvider(self.provider)
            self.algorithms_loaded = True
        except Exception as e:
            QgsApplication.messageLog().logMessage(f"Failed to initialize TrailScan processing provider: {e}",
                                                   "TrailScan", Qgis.Critical)
            self.algorithms_loaded = False

    def unload(self):
        """Remove toolbar and processing provider."""
        try:
            del self.toolbar
            if hasattr(self, 'provider') and self.provider:
                QgsApplication.processingRegistry().removeProvider(self.provider)
        except Exception as e:
            QgsApplication.messageLog().logMessage(f"Error unloading TrailScan plugin: {e}", "TrailScan")

    def runPreProcessing(self):
        """Run preprocessing algorithm."""
        if not self.algorithms_loaded:
            self.iface.messageBar().pushCritical("TrailScan Plugin",
                                                 "Processing algorithms not loaded.")
            return
        try:
            processing.execAlgorithmDialog("trailscan:preprocessing")
        except Exception as e:
            QgsApplication.messageLog().logMessage(f"Failed to run preprocessing: {e}", "TrailScan", Qgis.Critical)
            self.iface.messageBar().pushCritical("TrailScan Plugin", f"Failed to run preprocessing: {e}")

    def runInference(self):
        """Run inference algorithm."""
        if not self.algorithms_loaded:
            self.iface.messageBar().pushCritical("TrailScan Plugin",
                                                 "Processing algorithms not loaded.")
            return
        try:
            processing.execAlgorithmDialog("trailscan:inference")
        except Exception as e:
            QgsApplication.messageLog().logMessage(f"Failed to run inference: {e}", "TrailScan", Qgis.Critical)
            self.iface.messageBar().pushCritical("TrailScan Plugin", f"Failed to run inference: {e}")

# --------------------------------------------------------------------
class TrailScanProvider(QgsProcessingProvider):
    """Processing provider for TrailScan."""

    def __init__(self):
        super().__init__()

    def id(self):
        return 'trailscan'

    def name(self):
        return 'TrailScan'

    def description(self):
        return 'Deep learning-based segmentation of skid trails.'

    # Provider icon (appears at the top of the toolbox)
    def icon(self):
        return QIcon(self.svgIconPath())

    def svgIconPath(self):
        return os.path.join(os.path.dirname(__file__), 'TrailScan_Logo.svg')

    def loadAlgorithms(self):
        """Load Preprocessing and Inference algorithms."""
        try:
            from .preprocessing import TrailscanPreProcessingAlgorithm
            from .inference import TrailscanInferenceProcessingAlgorithm
        except ImportError:
            from preprocessing import TrailscanPreProcessingAlgorithm
            from inference import TrailscanInferenceProcessingAlgorithm

        self.addAlgorithm(TrailscanPreProcessingAlgorithm())
        self.addAlgorithm(TrailscanInferenceProcessingAlgorithm())
        QgsApplication.messageLog().logMessage("TrailScan: Algorithms loaded successfully", "TrailScan")

# --------------------------------------------------------------------
class TrailscanPreProcessingAlgorithm(QgsProcessingAlgorithm):
    """Preprocessing algorithm for TrailScan."""

    POINTCLOUD = "POINTCLOUD"
    OUTPUT_NORMALIZED = "OUTPUT_NORMALIZED"

    def name(self):
        return "preprocessing"

    def displayName(self):
        return "01 Preprocessing Point Cloud"

    def group(self):
        return ""

    def groupId(self):
        return ""

    def shortHelpString(self):
        return "Preprocess ALS point clouds to generate normalized multi-band raster (DTM, CHM, MRM, VDI)."

    # Algorithm icon for toolbox
    def icon(self):
        return QIcon(os.path.join(os.path.dirname(__file__), 'TrailScan_Logo.svg'))

    def initAlgorithm(self, config=None):
        from qgis.core import QgsProcessingParameterPointCloudLayer, QgsProcessingParameterRasterDestination
        self.addParameter(
            QgsProcessingParameterPointCloudLayer(self.POINTCLOUD, "Input point cloud")
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(self.OUTPUT_NORMALIZED, "Normalized Raster")
        )

    def createInstance(self):
        return self.__class__()

# --------------------------------------------------------------------
class TrailscanInferenceProcessingAlgorithm(QgsProcessingAlgorithm):
    """Inference algorithm for TrailScan."""

    INPUT_RASTER = "INPUT_RASTER"
    OUTPUT_TRAILMAP = "OUTPUT_TRAILMAP"

    def name(self):
        return "inference"

    def displayName(self):
        return "02 TrailScan Inference"

    def group(self):
        return ""

    def groupId(self):
        return ""

    def shortHelpString(self):
        return "Run TrailScan deep learning model on preprocessed raster to generate probability map of skid trails."

    # Algorithm icon for toolbox
    def icon(self):
        return QIcon(os.path.join(os.path.dirname(__file__), 'TrailScan_Inference.svg'))

    def initAlgorithm(self, config=None):
        from qgis.core import QgsProcessingParameterRasterLayer, QgsProcessingParameterRasterDestination
        self.addParameter(
            QgsProcessingParameterRasterLayer(self.INPUT_RASTER, "Normalized input raster")
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(self.OUTPUT_TRAILMAP, "Trailmap output")
        )

    def createInstance(self):
        return self.__class__()
