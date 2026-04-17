import sys
import subprocess
import os
from qgis.core import QgsMessageLog, Qgis

packages = None
packages_ready = True
current_dir = os.path.dirname(os.path.abspath(__file__))
file = os.path.join(current_dir, "Trailscan", "requirements.txt")

with open(file, 'r') as f:
    packages = [line.strip() for line in f.readlines()]


for p in packages:
    try:
        __import__(p)
        QgsMessageLog.logMessage(f"Package '{p}' is already installed.", "TrailScan", Qgis.Info)
    except ImportError:
        QgsMessageLog.logMessage(f"Error: Package '{p}' not found", "TrailScan", Qgis.Critical)
        packages_ready = False

if not packages_ready:
    QgsMessageLog.logMessage("Some required packages are missing. Please install the missing packages and restart QGIS.", "TrailScan", Qgis.Critical)
    sys.exit(1)

