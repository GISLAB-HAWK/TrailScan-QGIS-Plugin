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
from qgis.core import QgsApplication, QgsProcessingProvider
from .preprocessing import TrailscanPreProcessingAlgorithm
from .inference import TrailscanInferenceProcessingAlgorithm
import processing
import os

def classFactory(iface):
    return MinimalPlugin(iface)


class MinimalPlugin:
    def __init__(self, iface):
        self.iface = iface

    def initGui(self):
        self.toolbar = self.iface.addToolBar("Trailscan")
        self.action = QAction('Trailscan Preprocessing', self.iface.mainWindow())
        self.action2 = QAction('Trailscan Inference', self.iface.mainWindow())
        self.toolbar.addAction(self.action)
        self.toolbar.addAction(self.action2)
        self.action.triggered.connect(self.runPreProcessing)
        self.action2.triggered.connect(self.runInference)
        

        self.initProcessing()

    def initProcessing(self):
        self.provider = TrailScanProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def unload(self):
        del self.toolbar
        QgsApplication.processingRegistry().removeProvider(self.provider)

    def runPreProcessing(self):
        processing.execAlgorithmDialog("trailscan:preprocessing")

    def runInference(self):
        processing.execAlgorithmDialog("trailscan:inference")


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
        # Load algorithms here
        self.addAlgorithm(TrailscanPreProcessingAlgorithm())
        self.addAlgorithm(TrailscanInferenceProcessingAlgorithm())
