import pandas as pd
import numpy as np
import os
from database import DatabaseManager, init_database

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
    Load concrete strength dataset from database or create sample data
    """
    # Initialize database
    init_database()
    
    # Try to load from database first
    db_manager = DatabaseManager()
    try:
        df = db_manager.load_dataset()
        if df is not None and len(df) > 0:
            print(f"Loaded {len(df)} records from database")
            return df
    except Exception as e:
        print(f"Could not load from database: {e}")
    finally:
        db_manager.close()
    
    # Try to load from uploaded files
    try:
        if os.path.exists('attached_assets/Concrete_Data_1752784302540.xls'):
            df = pd.read_excel('attached_assets/Concrete_Data_1752784302540.xls')
            # Save to database for future use
            save_data_to_database(df, source='uploaded')
            return df
    except Exception as e:
        print(f"Could not load Excel file: {e}")
    
    # If no data found, create sample data and save to database
    print("Creating sample concrete dataset...")
    df = create_sample_concrete_data()
    save_data_to_database(df, source='synthetic')
    return df

def save_data_to_database(df, source='synthetic'):
    """
    Save dataframe to database
    """
    db_manager = DatabaseManager()
    try:
        success = db_manager.save_dataset(df, source=source)
        if success:
            print(f"Saved {len(df)} records to database")
        else:
            print("Failed to save data to database")
    except Exception as e:
        print(f"Error saving to database: {e}")
    finally:
        db_manager.close()

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
