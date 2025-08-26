#-----------------------------------------------------------
# Copyright (C) 2025 Tanja Kempen, Mathias Gröbe
#-----------------------------------------------------------
# Licensed under the terms of GNU GPL 2
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

#---------------------------------------------------------------------

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QAction, QMessageBox
from qgis.core import QgsApplication, QgsProcessingProvider, Qgis
import processing
import os
import sys
import subprocess
import importlib
import shutil

def classFactory(iface):
    return TrailScan(iface)

class TrailScan:
    def __init__(self, iface):
        self.iface = iface
        self.algorithms_loaded = False

    def initGui(self):
        self.toolbar = self.iface.addToolBar("Trailscan")
        self.toolbar.setObjectName("TrailScanToolbar")
        self.action = QAction('Trailscan Preprocessing', self.iface.mainWindow())
        self.action2 = QAction('Trailscan Inference', self.iface.mainWindow())
        self.toolbar.addAction(self.action)
        self.toolbar.addAction(self.action2)
        self.action.triggered.connect(self.runPreProcessing)
        self.action2.triggered.connect(self.runInference)

        self.initProcessing()

    def initProcessing(self):
        try:
            self.provider = TrailScanProvider()
            QgsApplication.processingRegistry().addProvider(self.provider)
            self.algorithms_loaded = True
        except Exception as e:
            QgsApplication.messageLog().logMessage(f"Failed to initialize TrailScan processing provider: {e}",
                                                   "TrailScan", Qgis.Critical)
            self.algorithms_loaded = False

    def unload(self):
        try:
            del self.toolbar
            if hasattr(self, 'provider') and self.provider:
                QgsApplication.processingRegistry().removeProvider(self.provider)
        except Exception as e:
            QgsApplication.messageLog().logMessage(f"Error unloading TrailScan plugin: {e}", "TrailScan")

    def runPreProcessing(self):
        if not self.algorithms_loaded:
            self.iface.messageBar().pushCritical("TrailScan Plugin",
                                                 "Processing algorithms not loaded. Please check package installation.")
            return

        try:
            processing.execAlgorithmDialog("trailscan:preprocessing")
        except Exception as e:
            QgsApplication.messageLog().logMessage(f"Failed to run preprocessing: {e}", "TrailScan", Qgis.Critical)
            self.iface.messageBar().pushCritical("TrailScan Plugin",
                                                 f"Failed to run preprocessing: {e}")

    def runInference(self):
        if not self.algorithms_loaded:
            self.iface.messageBar().pushCritical("TrailScan Plugin",
                                                 "Processing algorithms not loaded. Please check package installation.")
            return

        try:
            processing.execAlgorithmDialog("trailscan:inference")
        except Exception as e:
            QgsApplication.messageLog().logMessage(f"Failed to run inference: {e}", "TrailScan", Qgis.Critical)
            self.iface.messageBar().pushCritical("TrailScan Plugin",
                                                 f"Failed to run inference: {e}")


class TrailScanProvider(QgsProcessingProvider):
    def __init__(self):
        super().__init__()

    def id(self):
        return 'trailscan'

    def name(self):
        return 'TrailScan'

    def description(self):
        return 'A minimal plugin for trail scanning.'

    def icon(self):
        return QIcon(self.svgIconPath())

    def svgIconPath(self):
        return os.path.join(os.path.dirname(__file__), 'TrailScan_Logo.svg')

    def loadAlgorithms(self):
        # Load algorithms here - lazy loading to ensure packages are available
        try:
            # Try relative imports first
            try:
                from .preprocessing import TrailscanPreProcessingAlgorithm
                from .inference import TrailscanInferenceProcessingAlgorithm
            except ImportError:
                # Fallback to absolute imports
                from preprocessing import TrailscanPreProcessingAlgorithm
                from inference import TrailscanInferenceProcessingAlgorithm
            
            self.addAlgorithm(TrailscanPreProcessingAlgorithm())
            self.addAlgorithm(TrailscanInferenceProcessingAlgorithm())
            QgsApplication.messageLog().logMessage("TrailScan: Algorithms loaded successfully", "TrailScan")
        except ImportError as e:
            # Log the error but don't crash
            QgsApplication.messageLog().logMessage(f"Failed to load TrailScan algorithms: {e}", "TrailScan",
                                                   Qgis.Critical)
        except Exception as e:
            QgsApplication.messageLog().logMessage(f"Unexpected error loading TrailScan algorithms: {e}", "TrailScan",
                                                   Qgis.Critical)
