import pandas as pd
import numpy as np
import os

def create_sample_concrete_data():
    """
    Create sample concrete strength dataset based on the UCI concrete dataset structure
    """
    np.random.seed(42)  # For reproducibility
    
    # Generate sample data with realistic ranges based on the UCI dataset
    n_samples = 1000
    
    data = {
        'Cement': np.random.uniform(100, 600, n_samples),
        'BlastFurnaceSlag': np.random.uniform(0, 400, n_samples),
        'FlyAsh': np.random.uniform(0, 250, n_samples),
        'Water': np.random.uniform(120, 250, n_samples),
        'Superplasticizer': np.random.uniform(0, 35, n_samples),
        'CoarseAggregate': np.random.uniform(700, 1200, n_samples),
        'FineAggregate': np.random.uniform(500, 950, n_samples),
        'Age': np.random.randint(1, 365, n_samples)
    }
    
    # Create realistic concrete strength values based on input features
    # This is a simplified model for demonstration
    strength = (
        data['Cement'] * 0.1 +
        data['BlastFurnaceSlag'] * 0.08 +
        data['FlyAsh'] * 0.06 +
        data['Water'] * (-0.15) +
        data['Superplasticizer'] * 0.5 +
        data['CoarseAggregate'] * 0.01 +
        data['FineAggregate'] * 0.01 +
        np.log(data['Age'] + 1) * 5 +
        np.random.normal(0, 5, n_samples)  # Add some noise
    )
    
    # Ensure strength values are positive and realistic
    strength = np.clip(strength, 10, 100)
    data['ConcreteStrength'] = strength
    
    return pd.DataFrame(data)

def load_concrete_data():
    """
    Load concrete strength dataset
    """
    # Try to load from uploaded files first
    try:
        # Check if Excel file exists
        if os.path.exists('attached_assets/Concrete_Data_1752784302540.xls'):
            df = pd.read_excel('attached_assets/Concrete_Data_1752784302540.xls')
            return df
    except Exception as e:
        print(f"Could not load Excel file: {e}")
    
    # If no file found, create sample data
    print("Creating sample concrete dataset...")
    return create_sample_concrete_data()

def get_feature_names():
    """
    Return the feature names and their descriptions
    """
    feature_info = {
        'Cement': 'Cement Quantity (kg/m³)',
        'BlastFurnaceSlag': 'Blast Furnace Slag (kg/m³)',
        'FlyAsh': 'Fly Ash (kg/m³)',
        'Water': 'Water (kg/m³)',
        'Superplasticizer': 'Superplasticizer (kg/m³)',
        'CoarseAggregate': 'Coarse Aggregate (kg/m³)',
        'FineAggregate': 'Fine Aggregate (kg/m³)',
        'Age': 'Age (days)'
    }
    return feature_info

def get_feature_ranges():
    """
    Return typical ranges for each feature
    """
    ranges = {
        'Cement': (100, 600),
        'BlastFurnaceSlag': (0, 400),
        'FlyAsh': (0, 250),
        'Water': (120, 250),
        'Superplasticizer': (0, 35),
        'CoarseAggregate': (700, 1200),
        'FineAggregate': (500, 950),
        'Age': (1, 365)
    }
    return ranges
