"""
Standalone Concrete Strength Prediction App
No database required - runs entirely offline
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime

# Import our ML components
from utilities import normalize_features, eval_cost, eval_gradient, predict_strength
from sklearn.model_selection import train_test_split

# Configure page
st.set_page_config(
    page_title="Concrete Strength Predictor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SimpleConcreteModel:
    """Simplified model without database dependencies"""
    
    def __init__(self):
        self.weights = None
        self.bias = None
        self.feature_mean = None
        self.feature_std = None
        self.cost_history = []
        self.is_trained = False
        self.training_results = {}
    
    def prepare_data(self, df):
        """Prepare data for training"""
        feature_columns = ['Cement', 'BlastFurnaceSlag', 'FlyAsh', 'Water', 
                          'Superplasticizer', 'CoarseAggregate', 'FineAggregate', 'Age']
        X = df[feature_columns].values
        y = df['ConcreteStrength'].values
        return X, y
    
    def train(self, X, y, learning_rate=0.01, num_iterations=1000, test_size=0.2):
        """Train the model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Normalize features
        X_train_norm, self.feature_mean, self.feature_std = normalize_features(X_train)
        X_test_norm = (X_test - self.feature_mean) / self.feature_std
        
        # Initialize parameters
        n_features = X_train_norm.shape[1]
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0.0
        
        # Training loop
        self.cost_history = []
        for i in range(num_iterations):
            cost = eval_cost(X_train_norm, y_train, self.weights, self.bias)
            self.cost_history.append(cost)
            
            dJdw, dJdb = eval_gradient(X_train_norm, y_train, self.weights, self.bias)
            self.weights = self.weights - learning_rate * dJdw
            self.bias = self.bias - learning_rate * dJdb
        
        self.is_trained = True
        
        # Calculate results
        train_cost = eval_cost(X_train_norm, y_train, self.weights, self.bias)
        test_cost = eval_cost(X_test_norm, y_test, self.weights, self.bias)
        
        train_predictions = self.predict_batch(X_train)
        test_predictions = self.predict_batch(X_test)
        
        train_r2 = self.calculate_r2(y_train, train_predictions)
        test_r2 = self.calculate_r2(y_test, test_predictions)
        
        self.training_results = {
            'train_cost': train_cost,
            'test_cost': test_cost,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'train_predictions': train_predictions,
            'test_predictions': test_predictions
        }
        
        return self.training_results
    
    def predict(self, input_features):
        """Make a single prediction"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        return predict_strength(
            input_features, self.weights, self.bias, 
            self.feature_mean, self.feature_std
        )
    
    def predict_batch(self, X):
        """Predict for multiple samples"""
        predictions = []
        for i in range(X.shape[0]):
            pred = self.predict(X[i])
            predictions.append(pred)
        return np.array(predictions)
    
    def calculate_r2(self, y_true, y_pred):
        """Calculate R² score"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def save_model(self, filename):
        """Save model to file"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        model_data = {
            'weights': self.weights.tolist(),
            'bias': float(self.bias),
            'feature_mean': self.feature_mean.tolist(),
            'feature_std': self.feature_std.tolist(),
            'training_results': {
                'train_r2': float(self.training_results.get('train_r2', 0)),
                'test_r2': float(self.training_results.get('test_r2', 0)),
                'train_cost': float(self.training_results.get('train_cost', 0)),
                'test_cost': float(self.training_results.get('test_cost', 0))
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(model_data, f)
    
    def load_model(self, filename):
        """Load model from file"""
        try:
            with open(filename, 'r') as f:
                model_data = json.load(f)
            
            self.weights = np.array(model_data['weights'])
            self.bias = model_data['bias']
            self.feature_mean = np.array(model_data['feature_mean'])
            self.feature_std = np.array(model_data['feature_std'])
            self.training_results = model_data.get('training_results', {})
            self.is_trained = True
            return True
        except FileNotFoundError:
            return False

def create_sample_data():
    """Create sample concrete dataset"""
    np.random.seed(42)
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
    
    # Create realistic strength values
    strength = (
        data['Cement'] * 0.1 +
        data['BlastFurnaceSlag'] * 0.08 +
        data['FlyAsh'] * 0.06 +
        data['Water'] * (-0.15) +
        data['Superplasticizer'] * 0.5 +
        data['CoarseAggregate'] * 0.01 +
        data['FineAggregate'] * 0.01 +
        np.log(data['Age'] + 1) * 5 +
        np.random.normal(0, 5, n_samples)
    )
    
    strength = np.clip(strength, 10, 100)
    data['ConcreteStrength'] = strength
    
    return pd.DataFrame(data)

def get_feature_ranges():
    """Get feature ranges for sliders"""
    return {
        'Cement': (100, 600),
        'BlastFurnaceSlag': (0, 400),
        'FlyAsh': (0, 250),
        'Water': (120, 250),
        'Superplasticizer': (0, 35),
        'CoarseAggregate': (700, 1200),
        'FineAggregate': (500, 950),
        'Age': (1, 365)
    }

def get_feature_names():
    """Get feature descriptions"""
    return {
        'Cement': 'Cement Quantity (kg/m³)',
        'BlastFurnaceSlag': 'Blast Furnace Slag (kg/m³)',
        'FlyAsh': 'Fly Ash (kg/m³)',
        'Water': 'Water (kg/m³)',
        'Superplasticizer': 'Superplasticizer (kg/m³)',
        'CoarseAggregate': 'Coarse Aggregate (kg/m³)',
        'FineAggregate': 'Fine Aggregate (kg/m³)',
        'Age': 'Age (days)'
    }

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = SimpleConcreteModel()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

def load_data():
    """Load sample data"""
    if not st.session_state.data_loaded:
        with st.spinner("Creating sample concrete dataset..."):
            st.session_state.df = create_sample_data()
            st.session_state.data_loaded = True
    return st.session_state.df

def main():
    st.title("🏗️ Concrete Strength Predictor")
    st.markdown("### Offline Machine Learning Application")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "📊 Data Overview", 
        "🤖 Model Training", 
        "🔮 Prediction", 
        "📈 Model Performance",
        "📋 Prediction History"
    ])
    
    if page == "📊 Data Overview":
        show_data_overview()
    elif page == "🤖 Model Training":
        show_model_training()
    elif page == "🔮 Prediction":
        show_prediction_interface()
    elif page == "📈 Model Performance":
        show_model_performance()
    elif page == "📋 Prediction History":
        show_prediction_history()

def show_data_overview():
    st.header("📊 Dataset Overview")
    
    df = load_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Information")
        st.write(f"**Number of samples:** {len(df)}")
        st.write(f"**Number of features:** {len(df.columns) - 1}")
        st.write(f"**Target variable:** Concrete Strength (MPa)")
        
        st.subheader("Feature Descriptions")
        feature_info = get_feature_names()
        for feature, description in feature_info.items():
            st.write(f"• **{feature}:** {description}")
    
    with col2:
        st.subheader("Data Statistics")
        st.dataframe(df.describe())
    
    st.subheader("Feature Distributions")
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    n_cols = 3
    n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=numeric_columns,
        vertical_spacing=0.1
    )
    
    for i, col in enumerate(numeric_columns):
        row = i // n_cols + 1
        col_idx = i % n_cols + 1
        
        fig.add_trace(
            go.Histogram(x=df[col], name=col, showlegend=False),
            row=row, col=col_idx
        )
    
    fig.update_layout(height=300*n_rows, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def show_model_training():
    st.header("🤖 Model Training")
    
    df = load_data()
    
    # Training parameters
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Training Parameters")
        learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
        num_iterations = st.slider("Number of Iterations", 100, 5000, 1000, 100)
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training model..."):
                X, y = st.session_state.model.prepare_data(df)
                results = st.session_state.model.train(
                    X, y, learning_rate, num_iterations, test_size
                )
                st.session_state.model_trained = True
                st.success("Model trained successfully!")
                st.rerun()
    
    with col2:
        if st.session_state.model_trained:
            st.subheader("Training Results")
            results = st.session_state.model.training_results
            
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("Training R²", f"{results['train_r2']:.4f}")
                st.metric("Training Cost", f"{results['train_cost']:.4f}")
            with col2b:
                st.metric("Testing R²", f"{results['test_r2']:.4f}")
                st.metric("Testing Cost", f"{results['test_cost']:.4f}")

def show_prediction_interface():
    st.header("🔮 Concrete Strength Prediction")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first in the 'Model Training' section.")
        return
    
    st.markdown("### Input Concrete Mix Parameters")
    
    feature_ranges = get_feature_ranges()
    feature_info = get_feature_names()
    
    col1, col2 = st.columns(2)
    input_values = {}
    
    with col1:
        st.subheader("Cement Components")
        input_values['Cement'] = st.slider("Cement (kg/m³)", 
            float(feature_ranges['Cement'][0]), float(feature_ranges['Cement'][1]), 400.0, 10.0)
        input_values['BlastFurnaceSlag'] = st.slider("Blast Furnace Slag (kg/m³)", 
            float(feature_ranges['BlastFurnaceSlag'][0]), float(feature_ranges['BlastFurnaceSlag'][1]), 100.0, 10.0)
        input_values['FlyAsh'] = st.slider("Fly Ash (kg/m³)", 
            float(feature_ranges['FlyAsh'][0]), float(feature_ranges['FlyAsh'][1]), 50.0, 5.0)
        input_values['Water'] = st.slider("Water (kg/m³)", 
            float(feature_ranges['Water'][0]), float(feature_ranges['Water'][1]), 180.0, 5.0)
    
    with col2:
        st.subheader("Additives & Aggregates")
        input_values['Superplasticizer'] = st.slider("Superplasticizer (kg/m³)", 
            float(feature_ranges['Superplasticizer'][0]), float(feature_ranges['Superplasticizer'][1]), 10.0, 1.0)
        input_values['CoarseAggregate'] = st.slider("Coarse Aggregate (kg/m³)", 
            float(feature_ranges['CoarseAggregate'][0]), float(feature_ranges['CoarseAggregate'][1]), 1000.0, 50.0)
        input_values['FineAggregate'] = st.slider("Fine Aggregate (kg/m³)", 
            float(feature_ranges['FineAggregate'][0]), float(feature_ranges['FineAggregate'][1]), 750.0, 25.0)
        input_values['Age'] = st.slider("Age (days)", 
            int(feature_ranges['Age'][0]), int(feature_ranges['Age'][1]), 28, 1)
    
    # Make prediction
    input_array = np.array([
        input_values['Cement'], input_values['BlastFurnaceSlag'], input_values['FlyAsh'],
        input_values['Water'], input_values['Superplasticizer'], input_values['CoarseAggregate'],
        input_values['FineAggregate'], input_values['Age']
    ])
    
    prediction = st.session_state.model.predict(input_array)
    
    # Classify strength
    if prediction < 20:
        classification = "Low Strength"
        color = "🔴"
    elif prediction < 40:
        classification = "Medium Strength"
        color = "🟡"
    elif prediction < 60:
        classification = "High Strength"
        color = "🟢"
    else:
        classification = "Very High Strength"
        color = "🟢"
    
    st.markdown("---")
    st.subheader("🎯 Prediction Result")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.metric("Predicted Concrete Strength", f"{prediction:.2f} MPa")
        st.markdown(f"**Classification:** {color} {classification}")
    
    # Save prediction to history
    if st.button("💾 Save Prediction"):
        prediction_record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'inputs': input_values.copy(),
            'prediction': prediction,
            'classification': classification
        }
        st.session_state.prediction_history.append(prediction_record)
        st.success("Prediction saved to history!")

def show_model_performance():
    st.header("📈 Model Performance")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first.")
        return
    
    results = st.session_state.model.training_results
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Training R²", f"{results['train_r2']:.4f}")
    with col2:
        st.metric("Testing R²", f"{results['test_r2']:.4f}")
    with col3:
        st.metric("Training Cost", f"{results['train_cost']:.4f}")
    with col4:
        st.metric("Testing Cost", f"{results['test_cost']:.4f}")
    
    # Cost history
    st.subheader("Training Progress")
    cost_history = st.session_state.model.cost_history
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(cost_history))),
        y=cost_history,
        mode='lines',
        name='Training Cost'
    ))
    fig.update_layout(
        title="Cost Function During Training",
        xaxis_title="Iteration",
        yaxis_title="Cost",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Model save/load
    st.subheader("Model Management")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Save Model"):
            st.session_state.model.save_model('concrete_model.json')
            st.success("Model saved!")
    
    with col2:
        if st.button("📁 Load Model"):
            if st.session_state.model.load_model('concrete_model.json'):
                st.success("Model loaded!")
                st.session_state.model_trained = True
            else:
                st.error("No saved model found!")

def show_prediction_history():
    st.header("📋 Prediction History")
    
    history = st.session_state.prediction_history
    
    if not history:
        st.info("No predictions saved yet. Make some predictions first!")
        return
    
    st.subheader(f"Saved Predictions ({len(history)} records)")
    
    # Convert to DataFrame
    history_data = []
    for record in history:
        row = record['inputs'].copy()
        row['Predicted_Strength'] = record['prediction']
        row['Classification'] = record['classification']
        row['Timestamp'] = record['timestamp']
        history_data.append(row)
    
    history_df = pd.DataFrame(history_data)
    
    # Display summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Predictions", len(history))
    with col2:
        avg_strength = history_df['Predicted_Strength'].mean()
        st.metric("Average Strength", f"{avg_strength:.1f} MPa")
    with col3:
        max_strength = history_df['Predicted_Strength'].max()
        st.metric("Max Strength", f"{max_strength:.1f} MPa")
    
    # Show history table
    st.dataframe(history_df, use_container_width=True)
    
    # Download button
    if st.button("📥 Download History"):
        csv = history_df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )

if __name__ == "__main__":
    main()