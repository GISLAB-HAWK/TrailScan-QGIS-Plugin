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
    # Ensure stdout/stderr exist to avoid libraries writing to None in GUI context
    try:
        if getattr(sys, 'stderr', None) is None:
            sys.stderr = open(os.devnull, 'w')
        if getattr(sys, 'stdout', None) is None:
            sys.stdout = open(os.devnull, 'w')
    except Exception:
        pass

    print("TrailScan: Plugin initialization started")
    
    # Try to import and run package installer
    package_installer_available = False
    try:
        # Try multiple import methods
        try:
            from . import packages_installer_dialog
            print("TrailScan: Package installer imported successfully (relative import)")
        except ImportError:
            try:
                import packages_installer_dialog
                print("TrailScan: Package installer imported successfully (absolute import)")
            except ImportError:
                # Last resort: direct file import
                import importlib.util
                import os
                plugin_dir = os.path.dirname(os.path.abspath(__file__))
                spec = importlib.util.spec_from_file_location(
                    "packages_installer_dialog", 
                    os.path.join(plugin_dir, "packages_installer_dialog.py")
                )
                packages_installer_dialog = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(packages_installer_dialog)
                print("TrailScan: Package installer imported successfully (direct import)")
        
        packages_installer_dialog.check_required_packages_and_install_if_necessary(iface=iface)
        print("TrailScan: Package check completed")
        package_installer_available = True
    except ImportError as e:
        QgsApplication.messageLog().logMessage(f"Package installer import failed: {e}", "TrailScan", Qgis.Critical)
        print(f"TrailScan: Package installer import failed: {e}")
        print(f"TrailScan: Current working directory: {os.getcwd()}")
        print(f"TrailScan: Plugin directory: {os.path.dirname(os.path.abspath(__file__))}")
        print(f"TrailScan: Available files: {os.listdir(os.path.dirname(os.path.abspath(__file__)))}")
        
        # Additional debugging
        try:
            import sys
            print(f"TrailScan: Python executable: {sys.executable}")
            print(f"TrailScan: Python version: {sys.version}")
            print(f"TrailScan: Python path entries:")
            for i, path in enumerate(sys.path[:5]):
                print(f"  {i}: {path}")
        except Exception as debug_e:
            print(f"TrailScan: Debug info failed: {debug_e}")
        
        # Try to show a user-friendly message
        try:
            iface.messageBar().pushCritical("TrailScan Plugin", 
                                          f"Package installer not found: {e}. Plugin may not function correctly.")
        except:
            pass
    except Exception as e:
        QgsApplication.messageLog().logMessage(f"Package check failed: {e}", "TrailScan", Qgis.Critical)
        print(f"TrailScan: Package check failed in main initialization: {e}")
        import traceback
        traceback.print_exc()
        
        # Provide more specific error information
        error_msg = f"Package installation failed: {str(e)}"
        if "Permission denied" in str(e):
            error_msg += "\n\nThis might be due to insufficient permissions. Try running QGIS as administrator."
        elif "network" in str(e).lower() or "connection" in str(e).lower():
            error_msg += "\n\nThis might be due to network issues. Check your internet connection."
        elif "pip" in str(e).lower():
            error_msg += "\n\nThis might be due to pip installation issues. Try updating pip first."
        elif "import" in str(e).lower():
            error_msg += "\n\nThis might be due to missing dependencies. Check if all required packages are installed."
        
        # Try to show a user-friendly message
        try:
            iface.messageBar().pushWarning("TrailScan Plugin", error_msg)
        except:
            pass
    
    # Fallback package check if installer is not available
    if not package_installer_available:
        print("TrailScan: Running fallback package check...")
        try:
            # Simple package availability check
            required_packages = ['numpy', 'scipy', 'laspy', 'lazrs', 'rasterio', 'onnxruntime']
            missing_packages = []
            
            for package in required_packages:
                try:
                    __import__(package)
                    print(f"TrailScan: ✓ {package} is available")
                except ImportError:
                    print(f"TrailScan: ✗ {package} is missing")
                    missing_packages.append(package)
            
            if missing_packages:
                msg = f"Missing required packages: {', '.join(missing_packages)}. Plugin may not function correctly."
                QgsApplication.messageLog().logMessage(msg, "TrailScan", Qgis.Critical)
                try:
                    iface.messageBar().pushWarning("TrailScan Plugin", 
                                                  f"Missing packages: {', '.join(missing_packages)}. Install manually or check plugin installation.")
                except:
                    pass
            else:
                print("TrailScan: All required packages are available")
                
        except Exception as e:
            print(f"TrailScan: Fallback package check failed: {e}")
            QgsApplication.messageLog().logMessage(f"Fallback package check failed: {e}", "TrailScan", Qgis.Critical)
    
    # Check if PDAL CLI is available
    try:
        import subprocess
        result = subprocess.run(["pdal", "--version"], 
                              capture_output=True, text=True, 
                              timeout=10)
        if result.returncode != 0:
            raise RuntimeError("PDAL command failed")
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, RuntimeError):
        msg = ("TrailScan Plugin: PDAL CLI not found or not working. "
               "Please install PDAL system-wide and ensure it's on PATH. "
               "The plugin will not function without PDAL.")
        QgsApplication.messageLog().logMessage(msg, "TrailScan", Qgis.Critical)
        try:
            iface.messageBar().pushCritical("TrailScan Plugin", 
                                          "PDAL CLI not found. Please install PDAL system-wide.")
        except:
            pass
    
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
            QgsApplication.messageLog().logMessage(f"Failed to initialize TrailScan processing provider: {e}", "TrailScan", Qgis.Critical)
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
            from .preprocessing import TrailscanPreProcessingAlgorithm
            from .inference import TrailscanInferenceProcessingAlgorithm
            self.addAlgorithm(TrailscanPreProcessingAlgorithm())
            self.addAlgorithm(TrailscanInferenceProcessingAlgorithm())
            QgsApplication.messageLog().logMessage("TrailScan: Algorithms loaded successfully", "TrailScan")
        except ImportError as e:
            # Log the error but don't crash
            QgsApplication.messageLog().logMessage(f"Failed to load TrailScan algorithms: {e}", "TrailScan", Qgis.Critical)
        except Exception as e:
            QgsApplication.messageLog().logMessage(f"Unexpected error loading TrailScan algorithms: {e}", "TrailScan", Qgis.Critical)
