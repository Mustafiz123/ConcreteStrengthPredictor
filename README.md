# Concrete Strength Prediction App

A comprehensive machine learning application for predicting concrete compressive strength with database integration.

## Features

- **Web Application**: Interactive Streamlit interface
- **Machine Learning**: Custom linear regression from scratch
- **Database Integration**: PostgreSQL for data persistence
- **Prediction History**: Track and visualize past predictions
- **Model Management**: Save, load, and version ML models
- **Android Support**: Mobile app using Kivy/KivyMD

## Quick Start

### Option 1: Web Application (Recommended)

1. **Install Dependencies**
   ```bash
   pip install streamlit numpy pandas matplotlib plotly scikit-learn psycopg2-binary sqlalchemy alembic
   ```

2. **Run the App**
   ```bash
   streamlit run app.py --server.port 8501
   ```

3. **Open in Browser**
   Navigate to `http://localhost:8501`

### Option 2: Standalone Desktop Version

1. **Install Dependencies**
   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```

2. **Run Training Script**
   ```bash
   python standalone_app.py
   ```

### Option 3: Android App (Advanced)

1. **Install Build Tools**
   ```bash
   pip install buildozer kivy kivymd
   ```

2. **Build APK**
   ```bash
   buildozer android debug
   ```

## File Structure

```
concrete-strength-app/
├── app.py                 # Main Streamlit application
├── standalone_app.py      # Desktop version without database
├── android_app.py         # Mobile app (Kivy)
├── main.py               # Android app entry point
├── ml_model.py           # Machine learning model
├── data_loader.py        # Data handling
├── utilities.py          # ML utilities
├── database.py           # Database operations
├── buildozer.spec        # Android build configuration
└── README.md            # This file
```

## Usage

### Training a Model

1. Go to "Model Training" tab
2. Adjust learning rate, iterations, and test size
3. Click "Start Training"
4. View results in "Model Performance" tab

### Making Predictions

1. Go to "Prediction" tab
2. Adjust concrete mix parameters using sliders
3. Click "Predict Strength"
4. View classification and strength prediction

### Database Features

- **Database Management**: View saved models and dataset statistics
- **Prediction History**: Track all predictions with timestamps
- **Model Versioning**: Save and load different model versions

## Input Parameters

- **Cement**: 100-600 kg/m³
- **Blast Furnace Slag**: 0-400 kg/m³
- **Fly Ash**: 0-250 kg/m³
- **Water**: 120-250 kg/m³
- **Superplasticizer**: 0-35 kg/m³
- **Coarse Aggregate**: 700-1200 kg/m³
- **Fine Aggregate**: 500-950 kg/m³
- **Age**: 1-365 days

## Output

- **Predicted Strength**: MPa (Megapascals)
- **Classification**: Low/Medium/High/Very High Strength

## Technical Details

- **Algorithm**: Custom Linear Regression with Gradient Descent
- **Normalization**: Z-score normalization
- **Database**: PostgreSQL with SQLAlchemy
- **Frontend**: Streamlit with Plotly visualizations
- **Mobile**: Kivy/KivyMD for Android compatibility

## Requirements

- Python 3.7+
- PostgreSQL (for full features)
- Modern web browser (for Streamlit version)

## License

Educational use - Based on UCI concrete dataset structure