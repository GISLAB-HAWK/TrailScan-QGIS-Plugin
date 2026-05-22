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
import sys


class _SafeStream:
    """No-op fallback for sys.stdout / sys.stderr when QGIS has no console.

    On Windows, QGIS runs without a console, so sys.stdout and sys.stderr
    can be None. Libraries such as NumPy may call sys.stderr.write(), which
    then raises 'AttributeError: NoneType has no attribute write' and floods
    the QGIS log. Substituting a harmless no-op stream prevents that.
    """

    def write(self, *args, **kwargs):
        return 0

    def flush(self, *args, **kwargs):
        pass


if sys.stdout is None:
    sys.stdout = _SafeStream()
if sys.stderr is None:
    sys.stderr = _SafeStream()

from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QMessageBox
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
        # Defer package check to avoid blocking UI during plugin load
        try:
            from qgis.PyQt import QtCore
            QtCore.QTimer.singleShot(100, lambda: packages_installer_dialog.check_required_packages_and_install_if_necessary(iface=iface))
        except Exception:
            # If Qt not available for some reason, fall back to direct call
            packages_installer_dialog.check_required_packages_and_install_if_necessary(iface=iface)
    except ImportError:
        try:
            # Fallback to absolute import
            import packages_installer_dialog
            try:
                from qgis.PyQt import QtCore
                QtCore.QTimer.singleShot(100, lambda: packages_installer_dialog.check_required_packages_and_install_if_necessary(iface=iface))
            except Exception:
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