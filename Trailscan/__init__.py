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

from PyQt5.QtWidgets import QAction, QMessageBox
from qgis.core import QgsApplication, QgsProcessingProvider
from .preprocessing import TrailscanPreProcessingAlgorithm
from .inference import TrailscanInferenceProcessingAlgorithm
import processing

def classFactory(iface):
    return MinimalPlugin(iface)


class MinimalPlugin:
    def __init__(self, iface):
        self.iface = iface

    def initGui(self):
        self.action = QAction('Trailscan Preprocessing', self.iface.mainWindow())
        self.action.triggered.connect(self.runPreProcessing)
        self.iface.addToolBarIcon(self.action)

        self.initProcessing()

    def initProcessing(self):
        self.provider = TrailScanProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def unload(self):
        self.iface.removeToolBarIcon(self.action)
        del self.action
        QgsApplication.processingRegistry().removeProvider(self.provider)

    def runPreProcessing(self):
        processing.execAlgorithmDialog("trailscan:preprocessing")


class TrailScanProvider(QgsProcessingProvider):
    def __init__(self):
        super().__init__()

    def id(self):
        return 'trailscan'

    def name(self):
        return 'TrailScan'

    def description(self):
        return 'A minimal plugin for trail scanning.'

    # def icon(self):
    #     pass

    def loadAlgorithms(self):
        # Load algorithms here
        self.addAlgorithm(TrailscanPreProcessingAlgorithm())
        self.addAlgorithm(TrailscanInferenceProcessingAlgorithm())
