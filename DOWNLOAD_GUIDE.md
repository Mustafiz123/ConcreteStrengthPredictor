# 📥 How to Download and Run the Concrete Strength Prediction App

## Quick Start (Easiest Method)

### Option 1: Standalone Version (No Database)
This is the simplest way to get started:

1. **Download the files** (copy these to your computer):
   - `standalone_app.py`
   - `utilities.py`
   - `install.py`

2. **Install Python** (if not already installed):
   - Download from https://python.org
   - Make sure to check "Add Python to PATH" during installation

3. **Run the installer**:
   ```bash
   python install.py
   ```
   Choose option 1 (Basic installation)

4. **Start the app**:
   ```bash
   python -m streamlit run standalone_app.py
   ```

5. **Open your browser** to http://localhost:8501

### Option 2: Full Version (With Database)
For advanced features including prediction history and model management:

1. **Download all files**:
   - `app.py`
   - `ml_model.py` 
   - `data_loader.py`
   - `utilities.py`
   - `database.py`
   - `install.py`

2. **Run installer**:
   ```bash
   python install.py
   ```
   Choose option 2 (Full installation)

3. **Set up PostgreSQL** (optional - app will work with sample data):
   - Install PostgreSQL locally, or
   - Use a cloud database service

4. **Start the app**:
   ```bash
   python -m streamlit run app.py
   ```

### Option 3: Android App
For mobile devices:

1. **Download Android files**:
   - `android_app.py`
   - `main.py`
   - `buildozer.spec`
   - `utilities.py`
   - `ml_model.py`
   - `data_loader.py`

2. **Install development tools**:
   ```bash
   python install.py
   ```
   Choose option 3 (Android development)

3. **Build APK** (Linux/macOS only):
   ```bash
   buildozer android debug
   ```

## Manual Installation

If the installer doesn't work, install packages manually:

```bash
pip install streamlit numpy pandas matplotlib plotly scikit-learn
```

For full features, also install:
```bash
pip install psycopg2-binary sqlalchemy alembic
```

For Android development:
```bash
pip install kivy kivymd buildozer
```

## File Descriptions

- **standalone_app.py** - Complete offline app with prediction history
- **app.py** - Full web app with database integration
- **ml_model.py** - Machine learning model implementation
- **utilities.py** - Mathematical functions for ML training
- **database.py** - PostgreSQL database operations
- **android_app.py** - Mobile app interface
- **data_loader.py** - Data management functions

## Troubleshooting

### Common Issues:

1. **"Module not found"**: Run the installer or install packages manually
2. **"Port already in use"**: Change port with `--server.port 8502`
3. **Database connection errors**: Use standalone version instead
4. **Kivy installation fails**: Use conda instead of pip for Kivy

### Getting Help:

1. Check the README.md for detailed documentation
2. Ensure Python 3.7+ is installed
3. Try the standalone version first
4. Use virtual environments to avoid conflicts

## System Requirements

- **Python**: 3.7 or higher
- **RAM**: 2GB minimum, 4GB recommended
- **Storage**: 500MB for app + dependencies
- **OS**: Windows 10+, macOS 10.14+, or Linux
- **Browser**: Chrome, Firefox, Safari, or Edge

## Features Available

### Standalone Version:
- ✅ Machine learning training
- ✅ Concrete strength prediction
- ✅ Performance visualization
- ✅ Local prediction history
- ✅ Model save/load (JSON files)

### Full Version (with database):
- ✅ All standalone features
- ✅ PostgreSQL data persistence
- ✅ Advanced prediction history with charts
- ✅ Model versioning and management
- ✅ Dataset statistics and analysis
- ✅ Export functionality

### Android Version:
- ✅ Mobile-optimized interface
- ✅ Touch-friendly sliders and controls
- ✅ Offline prediction capability
- ✅ Model training on device

## Next Steps

1. Start with the standalone version to test functionality
2. Upgrade to full version if you need advanced features
3. Build Android app for mobile access
4. Customize the code for your specific use case