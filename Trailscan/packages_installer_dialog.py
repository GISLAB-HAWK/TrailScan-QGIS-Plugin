"""
This QGIS plugin requires some Python packages to be installed and available.
This tool allows to install them in a local directory, if they are not installed yet.

This file is based on code from the QGIS plugin "Deepness"
(https://github.com/PUTvision/qgis-plugin-deepness, licensed under the Apache License, Version 2.0).

Original copyright: 2021-2025 PUT vision, Przemyslaw Aszkowski
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Modifications have been made by Tanja Kempen in 2025:
 - Adapted for use in the TrailScan QGIS plugin
"""


import importlib
import logging
import os
import subprocess
import sys
import traceback
import urllib
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import List

from qgis.PyQt import QtCore, uic
from qgis.PyQt.QtCore import pyqtSignal
from qgis.PyQt.QtGui import QCloseEvent
from qgis.PyQt.QtWidgets import QDialog, QMessageBox, QTextBrowser

PLUGIN_NAME = 'TrailScan'

PYTHON_VERSION = sys.version_info
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLUGIN_ROOT_DIR = os.path.realpath(os.path.abspath(os.path.join(SCRIPT_DIR, '..')))
PACKAGES_INSTALL_DIR = os.path.join(PLUGIN_ROOT_DIR, f'python{PYTHON_VERSION.major}.{PYTHON_VERSION.minor}')


FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'packages_installer_dialog.ui'))

_ERROR_COLOR = '#ff0000'


@dataclass
class PackageToInstall:
    name: str
    version: str
    import_name: str  # name while importing package

    def __str__(self):
        return f'{self.name}{self.version}'


REQUIREMENTS_PATH = os.path.join(SCRIPT_DIR, 'requirements.txt')

try:
    with open(REQUIREMENTS_PATH, 'r') as f:
        raw_txt = f.read()
    
    libraries_versions = {}
    
    for line in raw_txt.split('\n'):
        if line.startswith('#') or not line.strip():
            continue

        line = line.split(';')[0]

        if '==' in line:
            lib, version = line.split('==')
            libraries_versions[lib] = '==' + version
        elif '>=' in line:
            lib, version = line.split('>=')
            libraries_versions[lib] = '>=' + version
        elif '<=' in line:
            lib, version = line.split('<=')
            libraries_versions[lib] = '<=' + version
        else:
            libraries_versions[line] = ''
            
    print(f"TrailScan: Loaded {len(libraries_versions)} packages from requirements.txt")
    
except Exception as e:
    print(f"TrailScan: Error reading requirements.txt: {e}")
    # Fallback to hardcoded packages
    libraries_versions = {
        'numpy': '',
        'scipy': '',
        'laspy': '',
        'lazrs': '',
        'rasterio': '',
        'onnxruntime': ''
    }
    print("TrailScan: Using fallback package list")


# Build install list from requirements (skip system tools like 'pdal')
REQ_IMPORT_MAP = {
    'numpy': 'numpy',
    'scipy': 'scipy',
    'laspy': 'laspy',
    'lazrs': 'lazrs',
    'rasterio': 'rasterio',
    'onnxruntime': 'onnxruntime',
}

packages_to_install = []
for lib, ver in libraries_versions.items():
    if lib.lower() == 'pdal':
        continue
    import_name = REQ_IMPORT_MAP.get(lib, lib)
    packages_to_install.append(PackageToInstall(name=lib, version=ver, import_name=import_name))

# Determine correct Python executable, especially on Windows/QGIS where sys.executable may be qgis-bin.exe

def _find_embedded_python_executable() -> str:
    exe = sys.executable or ""
    try:
        import qgis.core  # type: ignore
        prefix = qgis.core.QgsApplication.prefixPath()
        candidates = []
        if os.name == 'nt':
            # Typical QGIS layouts on Windows
            candidates.extend([
                os.path.join(prefix, 'bin', 'python.exe'),
                os.path.join(prefix, 'apps', 'Python39', 'python.exe'),
                os.path.join(prefix, 'apps', 'Python310', 'python.exe'),
                os.path.join(prefix, 'apps', 'Python311', 'python.exe'),
                os.path.join(os.path.dirname(exe), 'python.exe'),  # near qgis-bin.exe
            ])
        else:
            # On non-Windows, sys.executable is usually fine
            candidates.append(exe)
        for cand in candidates:
            if cand and os.path.exists(cand) and os.path.basename(cand).lower().startswith('python'):
                return cand
        # Fallbacks
        return exe
    except Exception:
        return exe

PYTHON_EXECUTABLE_PATH = _find_embedded_python_executable()
print(f"TrailScan: Using Python executable: {PYTHON_EXECUTABLE_PATH}")

# Windows-specific subprocess flag to avoid popping new windows
WINDOWS = (os.name == 'nt')
CREATIONFLAGS = getattr(subprocess, 'CREATE_NO_WINDOW', 0) if WINDOWS else 0


class PackagesInstallerDialog(QDialog, FORM_CLASS):
    """
    Dialog witch controls the installation process of packages.
    UI design defined in the `packages_installer_dialog.ui` file.
    """

    signal_log_line = pyqtSignal(str)  # we need to use signal because we cannot edit GUI from another thread

    INSTALLATION_IN_PROGRESS = False  # to make sure we will not start the installation twice

    def __init__(self, iface, parent=None):
        super(PackagesInstallerDialog, self).__init__(parent)
        self.setupUi(self)
        self.iface = iface
        self.tb = self.textBrowser_log  # type: QTextBrowser
        # Harden the log widget to avoid accidental interactions being treated by QGIS as data sources
        try:
            # Prevent opening links or accepting drops that could be interpreted by QGIS
            self.tb.setOpenExternalLinks(False)
            self.tb.setOpenLinks(False)
            self.tb.setAcceptDrops(False)
            self.setAcceptDrops(False)
        except Exception:
            # If running outside of full Qt env, ignore
            pass
        self._create_connections()
        self._setup_message()
        self.aborted = False
        self.thread = None

    def move_to_top(self):
        """ Move the window to the top.
        Although if installed from plugin manager, the plugin manager will move itself to the top anyway.
        """
        self.setWindowState((self.windowState() & ~QtCore.Qt.WindowMinimized) | QtCore.Qt.WindowActive)

        if sys.platform == "linux" or sys.platform == "linux2":
            pass
        elif sys.platform == "darwin":  # MacOS
            self.raise_()  # FIXME: this does not really work, the window is still behind the plugin manager
        elif sys.platform == "win32":
            self.activateWindow()
        else:
            raise Exception("Unsupported operating system!")

    def _create_connections(self):
        self.pushButton_close.clicked.connect(self.close)
        self.pushButton_install_packages.clicked.connect(self._run_packages_installation)
        self.pushButton_diagnostics.clicked.connect(self._show_diagnostics)
        self.signal_log_line.connect(self._log_line)

    def _show_diagnostics(self):
        """Show diagnostic information about the environment"""
        self.log('\n\n')
        self.log('=' * 60)
        self.log('<h3><b>Environment Diagnostics</b></h3>')
        
        diagnostics = get_environment_diagnostics()
        for line in diagnostics:
            self.log(line)
        
        self.log('\n' + '=' * 60)
        self.log('<b>If packages are missing, try the following:</b>')
        self.log('1. Check if you have internet connection')
        self.log('2. Try running QGIS as administrator')
        self.log('3. Update pip: <code>python -m pip install --upgrade pip</code>')
        self.log('4. Install packages manually in a terminal (do not paste into QGIS):')
        self.log('<code>python -m pip install --user numpy scipy laspy lazrs rasterio onnxruntime</code>')
        self.log('5. Check QGIS Python environment settings')

    def _log_line(self, txt):
        txt = txt \
            .replace('  ', '&nbsp;&nbsp;') \
            .replace('\n', '<br>')
        self.tb.append(txt)

    def log(self, txt):
        self.signal_log_line.emit(txt)

    def _setup_message(self) -> None:
          
        self.log(f'<h2><span style="color: #000080;"><strong>  '
                 f'Plugin {PLUGIN_NAME} - Packages installer </strong></span></h2> \n'
                 f'\n'
                 f'<b>This plugin requires the following Python packages to be installed:</b>')
        
        for package in packages_to_install:
            self.log(f'\t- {package.name}{package.version}')

        self.log('\n\n'
                 f'If this packages are not installed in the global environment '
                 f'(or environment in which QGIS is started) '
                 f'you can install these packages in the local directory (which is included to the Python path).\n\n'
                 f'This Dialog does it for you! (Though you can still install these packages manually instead).\n'
                 f'<b>Please click "Install packages" button below to install them automatically, </b>'
                 f'or "Test and Close" if you installed them manually...\n')

    def _run_packages_installation(self):
        if self.INSTALLATION_IN_PROGRESS:
            self.log(f'Error! Installation already in progress, cannot start again!')
            return
        self.aborted = False
        self.INSTALLATION_IN_PROGRESS = True
        self.thread = Thread(target=self._install_packages)
        self.thread.start()

    def _install_packages(self) -> None:
        self.log('\n\n')
        self.log('=' * 60)
        self.log(f'<h3><b>Attempting to install required packages...</b></h3>')
        os.makedirs(PACKAGES_INSTALL_DIR, exist_ok=True)

        if PACKAGES_INSTALL_DIR not in sys.path:
            try:
                sys.path.insert(0, PACKAGES_INSTALL_DIR)
                self.log(f'Added {PACKAGES_INSTALL_DIR} to sys.path')
            except Exception:
                pass

        self._install_pip_if_necessary()

        self.log(f'<h3><b>Attempting to install required packages...</b></h3>\n')
        try:
            self._pip_install_packages(packages_to_install)
        except Exception as e:
            msg = (f'\n <span style="color: {_ERROR_COLOR};"><b> '
                   f'Packages installation failed with exception: {e}!\n'
                   f'Please try to install the packages again. </b></span>'
                   f'\nCheck if there is no error related to system packages, '
                   f'which may be required to be installed by your system package manager, e.g. "apt". '
                   f'Copy errors from the stack above and google for possible solutions. '
                   f'Please report these as an issue on the plugin repository tracker!')
            self.log(msg)

        # finally, validate the installation, if there was no error so far...
        self.log('\n\n <b>Installation of required packages finished. Validating installation...</b>')
        self._check_packages_installation_and_log()
        self.INSTALLATION_IN_PROGRESS = False

    def reject(self) -> None:
        self.close()

    def closeEvent(self, event: QCloseEvent):
        self.aborted = True
        if self._check_packages_installation_and_log():
            event.accept()
            return

        res = QMessageBox.question(self.iface.mainWindow(),
                                   f'{PLUGIN_NAME} - skip installation?',
                                   'Are you sure you want to abort the installation of the required python packages? '
                                   'The plugin may not function correctly without them!',
                                   QMessageBox.No, QMessageBox.Yes)
        log_msg = 'User requested to close the dialog, but the packages are not installed correctly!\n'
        if res == QMessageBox.Yes:
            log_msg += 'And the user confirmed to close the dialog, knowing the risk!'
            event.accept()
        else:
            log_msg += 'The user reconsidered their decision, and will try to install the packages again!'
            event.ignore()
        log_msg += '\n'
        self.log(log_msg)

    def _install_pip_if_necessary(self):
        """
        Install pip if not present.
        It happens e.g. in flatpak applications.

        TODO - investigate whether we can also install pip in local directory
        """

        self.log(f'<h4><b>Making sure pip is installed...</b></h4>')
        if check_pip_installed():
            self.log(f'<em>Pip is installed, skipping installation...</em>\n')
            return

        install_pip_command = [PYTHON_EXECUTABLE_PATH, '-m', 'ensurepip']
        self.log(f'<em>Running command to install pip: \n  $ {" ".join(install_pip_command)} </em>')
        with subprocess.Popen(install_pip_command,
                              stdout=subprocess.PIPE,
                              universal_newlines=True,
                              stderr=subprocess.STDOUT,
                              creationflags=CREATIONFLAGS,
                              env={'SETUPTOOLS_USE_DISTUTILS': 'stdlib'}) as process:
            try:
                self._do_process_output_logging(process)
            except InterruptedError as e:
                self.log(str(e))
                return False

        if process.returncode != 0:
            msg = (f'<span style="color: {_ERROR_COLOR};"><b>'
                   f'pip installation failed! Consider installing it manually.'
                   f'<b></span>')
            self.log(msg)
        self.log('\n')

    def _pip_install_packages(self, packages: List[PackageToInstall]) -> None:
        # Try to install packages one by one to avoid conflicts
        for pck in packages:
            try:
                # Skip installation if package already importable
                try:
                    importlib.import_module(pck.import_name)
                    self.log(f'<em>Skipping {pck.name}: already importable.</em>')
                    continue
                except Exception:
                    pass

                cmd = [
                    PYTHON_EXECUTABLE_PATH, '-m', 'pip', 'install',
                    '--no-input', '--upgrade', '--disable-pip-version-check', '--no-cache-dir',
                    '--timeout', '60', '--retries', '3',
                    f'--target={PACKAGES_INSTALL_DIR}',
                    f"{pck}"
                ]
                cmd_string = ' '.join(cmd)

                self.log(f'<em>Installing {pck.name}... Running command: \n  $ {cmd_string} </em>')

                # Prepare environment with PYTHONPATH including target dir
                env = os.environ.copy()
                existing_pp = env.get('PYTHONPATH', '')
                pp_parts = [PACKAGES_INSTALL_DIR] + ([existing_pp] if existing_pp else [])
                env['PYTHONPATH'] = os.pathsep.join([p for p in pp_parts if p])

                # Add timeout protection to prevent hanging
                try:
                    with subprocess.Popen(cmd,
                                          stdout=subprocess.PIPE,
                                          universal_newlines=True,
                                          stderr=subprocess.STDOUT,
                                          creationflags=CREATIONFLAGS,
                                          env=env) as process:
                        # Set a reasonable timeout (15 minutes)
                        try:
                            self._do_process_output_logging(process)
                            # Wait for process with timeout
                            process.wait(timeout=900)  # 15 minutes timeout
                        except subprocess.TimeoutExpired:
                            self.log(f'<span style="color: {_ERROR_COLOR};"><b>Timeout installing {pck.name} - killing process</b></span>')
                            process.kill()
                            process.wait()
                            continue
                        except Exception as e:
                            self.log(f'<span style="color: {_ERROR_COLOR};"><b>Error during {pck.name} installation: {e}</b></span>')
                            if process.poll() is None:
                                process.kill()
                                process.wait()
                            continue

                except Exception as subprocess_error:
                    self.log(f'<span style="color: {_ERROR_COLOR};"><b>Subprocess error installing {pck.name}: {subprocess_error}</b></span>')
                    continue

                if process.returncode != 0:
                    self.log(f'<span style="color: {_ERROR_COLOR};"><b>Failed to install {pck.name}</b></span>')
                    continue
                else:
                    self.log(f'<span style="color: #008000;"><b>Successfully installed {pck.name}</b></span>')
                    
            except Exception as e:
                self.log(f'<span style="color: {_ERROR_COLOR};"><b>Error installing {pck.name}: {e}</b></span>')
                continue

        msg = (f'\n<b>Package installation completed. Check the log above for details.</b>\n\n')
        self.log(msg)

    def _do_process_output_logging(self, process: subprocess.Popen) -> None:
        """
        :param process: instance of 'subprocess.Popen'
        """
        for stdout_line in iter(process.stdout.readline, ""):
            if stdout_line.isspace():
                continue
            txt = f'<span style="color: #999999;">{stdout_line.rstrip(os.linesep)}</span>'
            self.log(txt)
            if self.aborted:
                raise InterruptedError('Installation aborted by user')

    def _check_packages_installation_and_log(self) -> bool:
        packages_ok = are_packages_importable()
        self.pushButton_install_packages.setEnabled(not packages_ok)

        if packages_ok:
            msg1 = f'All required packages are importable! You can close this window now!'
            self.log(msg1)
            return True

        try:
            import_packages()
            raise Exception("Unexpected successful import of packages?!? It failed a moment ago, we shouldn't be here!")
        except Exception:
            msg_base = '<b>Python packages required by the plugin could not be loaded due to the following error:</b>'
            logging.exception(msg_base)
            tb = traceback.format_exc()
            msg1 = (f'<span style="color: {_ERROR_COLOR};">'
                    f'{msg_base} \n '
                    f'{tb}\n\n'
                    f'<b>Please try installing the packages again.<b>'
                    f'</span>')
            self.log(msg1)

        return False


dialog = None


def import_package(package: PackageToInstall):
    importlib.import_module(package.import_name)


def import_packages():
    for package in packages_to_install:
        import_package(package)


def are_packages_importable() -> bool:
    try:
        import_packages()
    except Exception:
        logging.exception(f'Python packages required by the plugin could not be loaded due to the following error:')
        return False

    return True


def check_qgis_python_packages():
    """Check if packages are available in QGIS Python environment"""
    try:
        # Try to import packages in QGIS context
        import qgis.core
        qgis.core.QgsApplication.messageLog().logMessage("TrailScan: Checking packages in QGIS Python environment", "TrailScan")
        
        if are_packages_importable():
            qgis.core.QgsApplication.messageLog().logMessage("TrailScan: All packages available in QGIS Python", "TrailScan")
            return True
        else:
            qgis.core.QgsApplication.messageLog().logMessage("TrailScan: Some packages missing in QGIS Python", "TrailScan")
            return False
    except Exception as e:
        print(f"TrailScan: Error checking QGIS Python packages: {e}")
        return False


def check_pip_installed() -> bool:
    try:
        res = subprocess.run([PYTHON_EXECUTABLE_PATH, '-m', 'pip', '--version'],
                             capture_output=True, text=True, creationflags=CREATIONFLAGS)
        return res.returncode == 0
    except Exception:
        return False


def _try_system_installation() -> bool:
    """Try to install packages using system pip"""
    try:
        print("TrailScan: Attempting to install packages using system pip...")
        
        # Install packages one by one to avoid conflicts
        success_count = 0
        for pck in packages_to_install:
            try:
                cmd = [PYTHON_EXECUTABLE_PATH, '-m', 'pip', 'install', '--user', f"{pck}"]
                print(f"TrailScan: Installing {pck.name}...")
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, creationflags=CREATIONFLAGS)
                if result.returncode == 0:
                    print(f"TrailScan: Successfully installed {pck.name}")
                    success_count += 1
                else:
                    print(f"TrailScan: Failed to install {pck.name}: {result.stderr}")
                    # Continue with other packages instead of failing completely
                    
            except subprocess.TimeoutExpired:
                print(f"TrailScan: Timeout installing {pck.name} - killing any hanging processes")
                # Try to kill any hanging pip processes
                try:
                    subprocess.run(['taskkill', '/f', '/im', 'python.exe'], 
                                  capture_output=True, timeout=10, creationflags=CREATIONFLAGS)
                except:
                    pass
                # Continue with other packages
            except Exception as e:
                print(f"TrailScan: Error installing {pck.name}: {e}")
                # Continue with other packages
        
        print(f"TrailScan: Successfully installed {success_count}/{len(packages_to_install)} packages")
        
        # Verify if we have enough packages to function
        if are_packages_importable():
            print("TrailScan: All packages successfully installed and importable")
            return True
        else:
            print("TrailScan: Some packages installed but not all importable")
            # Return True if we have at least some critical packages
            return success_count > 0
            
    except Exception as e:
        print(f"TrailScan: System installation failed: {e}")
        return False


def check_required_packages_and_install_if_necessary(iface):
    # Prevent multiple simultaneous installations
    if hasattr(check_required_packages_and_install_if_necessary, '_running'):
        print("TrailScan: Package installation already in progress, skipping...")
        return
    
    check_required_packages_and_install_if_necessary._running = True
    
    try:
        print(f"TrailScan: Checking required packages...")
        
        # Debug Python paths to help diagnose issues
        debug_python_paths()
        
        print(f"TrailScan: Python executable: {PYTHON_EXECUTABLE_PATH}")
        print(f"TrailScan: Packages install dir: {PACKAGES_INSTALL_DIR}")
        print(f"TrailScan: Requirements path: {REQUIREMENTS_PATH}")
        
        # First try to import packages from system
        if are_packages_importable():
            print("TrailScan: All packages are already available in system")
            return
        
        print("TrailScan: Some packages not available in system, attempting automatic installation...")
        
        # Try to install packages using system pip first
        if check_pip_installed():
            print("TrailScan: Pip is available, attempting system installation...")
            try:
                if _try_system_installation():
                    print("TrailScan: System installation successful!")
                    return
                else:
                    print("TrailScan: System installation failed, falling back to local installation...")
            except Exception as e:
                print(f"TrailScan: System installation error: {e}, falling back to local installation...")
        
        # If system installation fails, try local installation
        print("TrailScan: Attempting local installation...")
        os.makedirs(PACKAGES_INSTALL_DIR, exist_ok=True)
        if PACKAGES_INSTALL_DIR not in sys.path:
            sys.path.append(PACKAGES_INSTALL_DIR)
            print(f"TrailScan: Added {PACKAGES_INSTALL_DIR} to Python path")
        
        # Check again after adding local path
        if are_packages_importable():
            print("TrailScan: All packages are available in local installation")
            return
            
        print("TrailScan: Automatic installation failed, showing installer dialog...")
        
        # If still not available, show installer dialog
        global dialog
        dialog = PackagesInstallerDialog(iface)
        dialog.setWindowModality(QtCore.Qt.WindowModal)
        dialog.show()
        dialog.move_to_top()
        
    except Exception as e:
        # Log error but don't crash the plugin
        print(f"TrailScan: Package check failed: {e}")
        try:
            iface.messageBar().pushWarning("TrailScan Plugin", 
                                          f"Package check failed: {e}. Some features may not work.")
        except:
            pass
    finally:
        # Always clear the running flag
        check_required_packages_and_install_if_necessary._running = False


def get_environment_diagnostics():
    """Get diagnostic information about the Python environment"""
    diagnostics = []
    
    try:
        # Python version info
        diagnostics.append(f"Python Version: {sys.version}")
        diagnostics.append(f"Python Executable: {sys.executable}")
        diagnostics.append(f"Python Path: {sys.path[:3]}...")  # First 3 entries
        
        # Check pip
        try:
            import pip
            diagnostics.append(f"Pip Version: {pip.__version__}")
        except ImportError:
            diagnostics.append("Pip: Not available")
        
        # Check if we're in QGIS
        try:
            import qgis.core
            diagnostics.append("QGIS: Available")
            diagnostics.append(f"QGIS Version: {qgis.core.Qgis.QGIS_VERSION}")
        except ImportError:
            diagnostics.append("QGIS: Not available")
        
        # Check package import status
        diagnostics.append("\nPackage Status:")
        for package in packages_to_install:
            try:
                importlib.import_module(package.import_name)
                diagnostics.append(f"  {package.name}: ✓ Available")
            except ImportError:
                diagnostics.append(f"  {package.name}: ✗ Missing")
        
        # Check write permissions
        try:
            test_file = os.path.join(PACKAGES_INSTALL_DIR, "test_write.tmp")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            diagnostics.append(f"Write Permissions: ✓ Can write to {PACKAGES_INSTALL_DIR}")
        except Exception as e:
            diagnostics.append(f"Write Permissions: ✗ Cannot write to {PACKAGES_INSTALL_DIR} - {e}")
            
    except Exception as e:
        diagnostics.append(f"Error getting diagnostics: {e}")
    
    return diagnostics


def debug_python_paths():
    """Debug function to help diagnose Python path issues"""
    print("=" * 60)
    print("PYTHON PATH DEBUGGING")
    print("=" * 60)
    
    print(f"sys.executable: {sys.executable}")
    print(f"PYTHON_EXECUTABLE_PATH: {PYTHON_EXECUTABLE_PATH}")
    print(f"os.path.exists(PYTHON_EXECUTABLE_PATH): {os.path.exists(PYTHON_EXECUTABLE_PATH) if PYTHON_EXECUTABLE_PATH else 'None'}")
    
    # Check if we're in QGIS
    try:
        import qgis.core
        print(f"QGIS Python: {qgis.core.QgsApplication.prefixPath()}")
    except ImportError:
        print("QGIS not available")
    
    # Check common Python locations
    common_paths = [
        r"C:\Program Files\QGIS*\bin\python.exe",
        r"C:\Program Files\QGIS*\apps\Python*\python.exe",
        r"C:\OSGeo4W64\bin\python.exe",
        r"C:\OSGeo4W64\apps\Python*\python.exe",
    ]
    
    print("\nChecking common Python locations:")
    for path_pattern in common_paths:
        try:
            import glob
            matches = glob.glob(path_pattern)
            for match in matches:
                print(f"  {match}: {os.path.exists(match)}")
        except Exception as e:
            print(f"  {path_pattern}: Error - {e}")
    
    # Test PATH-based Python
    print("\nTesting PATH-based Python:")
    try:
        result = subprocess.run(["python", "--version"], capture_output=True, text=True, timeout=5, creationflags=CREATIONFLAGS)
        print(f"  python: {result.stdout.strip() if result.returncode == 0 else 'Not found'}")
    except Exception as e:
        print(f"  python: Error - {e}")
    
    try:
        result = subprocess.run(["python3", "--version"], capture_output=True, text=True, timeout=5, creationflags=CREATIONFLAGS)
        print(f"  python3: {result.stdout.strip() if result.returncode == 0 else 'Not found'}")
    except Exception as e:
        print(f"  python3: Error - {e}")
    
    print("=" * 60)
