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
    try:
        # Try to import packages_installer_dialog using relative import
        from . import packages_installer_dialog
        packages_installer_dialog.check_required_packages_and_install_if_necessary(iface=iface)
    except ImportError:
        try:
            # Fallback to absolute import
            import packages_installer_dialog
            packages_installer_dialog.check_required_packages_and_install_if_necessary(iface=iface)
        except ImportError as e:
            # If both fail, log error but continue
            print(f"Warning: Could not import packages_installer_dialog: {e}")
            print("Continuing without package installation check...")

    try:
        # Try relative import first
        from .trailscan import TrailScan
    except ImportError:
        # Fallback to absolute import
        from trailscan import TrailScan
    
    return TrailScan(iface)