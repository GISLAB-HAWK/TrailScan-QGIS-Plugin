# -----------------------------------------------------------
# Copyright (C) 2025 Tanja Kempen, Mathias Gröbe
# -----------------------------------------------------------
# Licensed under the terms of GNU GPL 2
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# ---------------------------------------------------------------------
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QAction, QMessageBox
from qgis.core import QgsApplication, QgsProcessingProvider, Qgis, QgsMessageLog
import processing
import os
import sys
import subprocess
import importlib
import shutil
import platform

# Global flag to prevent multiple package checks
_package_check_done = False


def _try_import_installer():
    """Try to import the packages_installer_dialog module (relative or absolute)."""
    try:
        from . import packages_installer_dialog
        installer = packages_installer_dialog
    except ImportError:
        try:
            import packages_installer_dialog
            installer = packages_installer_dialog
        except ImportError as e:
            QgsMessageLog.logMessage(
                f"Warning: Could not import packages_installer_dialog: {e}",
                "TrailScan",
                Qgis.Warning
            )
            QgsMessageLog.logMessage(
                "Continuing without package installation check...",
                "TrailScan",
                Qgis.Warning
            )
            installer = None

    # --- Log OS and Import-Status ---
    current_os = platform.system()
    QgsMessageLog.logMessage(
        f"Operating System detected: {current_os}",
        "TrailScan",
        Qgis.Info
    )
    if installer:
        QgsMessageLog.logMessage(
            "packages_installer_dialog successfully imported.",
            "TrailScan",
            Qgis.Info
        )
    else:
        QgsMessageLog.logMessage(
            "No packages_installer_dialog available; skipping package check.",
            "TrailScan",
            Qgis.Warning
        )

    return installer


def classFactory(iface):
    """Load TrailScan class from file trailscan.py and return its instance."""
    global _package_check_done

    if not _package_check_done:
        _package_check_done = True  # Prevent double initialization

        installer = _try_import_installer()
        if installer:
            try:
                from qgis.PyQt import QtCore
                # Timer für MacOS / Verzögerung vor Paket-Check
                QtCore.QTimer.singleShot(
                    2000,
                    lambda: installer.check_required_packages_and_install_if_necessary(iface=iface)
                )
                QgsMessageLog.logMessage(
                    "Package check scheduled via QTimer (2s delay) for MacOS.",
                    "TrailScan",
                    Qgis.Info
                )
            except Exception as e:
                QgsMessageLog.logMessage(
                    f"QTimer failed; running installer immediately: {e}",
                    "TrailScan",
                    Qgis.Warning
                )
                installer.check_required_packages_and_install_if_necessary(iface=iface)

    try:
        from .trailscan import TrailScan
    except ImportError:
        from trailscan import TrailScan

    return TrailScan(iface)

