"""
install_dependencies.py

Installs required Python libraries for the Demand Forecasting app and verifies imports.
"""
import subprocess
import sys
import importlib

# List of pip package names and corresponding modules to import
pip_packages = [
    "streamlit",
    "pandas",
    "numpy",
    "matplotlib",
    "statsmodels",
    "scikit-learn",
    "openpyxl",
    "xlsxwriter"
]

modules = [
    "streamlit",
    "pandas",
    "numpy",
    "matplotlib",
    "statsmodels",
    "sklearn",
    "openpyxl",
    "xlsxwriter"
]

def install_package(package_name):
    """Install a package via pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

def main():
    for pkg_name, module_name in zip(pip_packages, modules):
        try:
            importlib.import_module(module_name)
            print(f"✅ Module '{module_name}' is already installed.")
        except ImportError:
            print(f"⏳ Installing '{pkg_name}' ...")
            install_package(pkg_name)
            print(f"✅ Installed '{pkg_name}'.")
    print("\n🔍 Verifying installations:")
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"✅ Successfully imported '{module_name}' (version {version}).")
        except ImportError as e:
            print(f"❌ Failed to import '{module_name}': {e}")

if __name__ == "__main__":
    main()
