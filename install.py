#!/usr/bin/env python3
"""
Easy installer for Concrete Strength Prediction App
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("🏗️ Concrete Strength Prediction App Installer")
    print("=" * 50)
    
    # Required packages for basic functionality
    basic_packages = [
        "streamlit",
        "numpy",
        "pandas", 
        "matplotlib",
        "plotly",
        "scikit-learn"
    ]
    
    # Optional packages for full features
    advanced_packages = [
        "psycopg2-binary",
        "sqlalchemy",
        "alembic"
    ]
    
    print("Installing basic packages...")
    for package in basic_packages:
        print(f"Installing {package}...")
        if install_package(package):
            print(f"✓ {package} installed successfully")
        else:
            print(f"✗ Failed to install {package}")
    
    print("\nChoose installation type:")
    print("1. Basic (offline version, no database)")
    print("2. Full (with database features)")
    print("3. Android development (includes Kivy)")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "2":
        print("\nInstalling database packages...")
        for package in advanced_packages:
            print(f"Installing {package}...")
            if install_package(package):
                print(f"✓ {package} installed successfully")
            else:
                print(f"✗ Failed to install {package}")
    
    elif choice == "3":
        print("\nInstalling Android development packages...")
        android_packages = ["kivy", "kivymd", "buildozer"]
        for package in android_packages:
            print(f"Installing {package}...")
            if install_package(package):
                print(f"✓ {package} installed successfully")
            else:
                print(f"✗ Failed to install {package}")
    
    print("\n🎉 Installation complete!")
    print("\nTo run the app:")
    
    if choice == "1":
        print("  python -m streamlit run standalone_app.py")
    else:
        print("  python -m streamlit run app.py")
    
    print("\nThe app will open in your web browser at http://localhost:8501")

if __name__ == "__main__":
    main()