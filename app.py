import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_loader import load_concrete_data, get_feature_names, get_feature_ranges
from ml_model import ConcreteStrengthModel

# Configure page
st.set_page_config(
    page_title="Concrete Strength Prediction",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = ConcreteStrengthModel()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

def load_data():
    """Load and cache the concrete dataset"""
    if not st.session_state.data_loaded:
        with st.spinner("Loading concrete strength dataset..."):
            st.session_state.df = load_concrete_data()
            st.session_state.data_loaded = True
    return st.session_state.df

def train_model():
    """Train the machine learning model"""
    df = load_data()
    
    with st.spinner("Training machine learning model..."):
        X, y = st.session_state.model.prepare_data(df)
        
        # Get training parameters from sidebar
        learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
        num_iterations = st.sidebar.slider("Number of Iterations", 100, 5000, 1000, 100)
        test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
        
        # Train the model
        training_results = st.session_state.model.train(
            X, y, 
            learning_rate=learning_rate, 
            num_iterations=num_iterations,
            test_size=test_size
        )
        
        st.session_state.model_trained = True
        st.session_state.training_results = training_results
        
        st.success("Model trained successfully!")
        return training_results

def main():
    st.title("🏗️ Concrete Strength Prediction")
    st.markdown("### Machine Learning Application for Predicting Concrete Compressive Strength")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "📊 Data Overview", 
        "🤖 Model Training", 
        "🔮 Prediction", 
        "📈 Model Performance"
    ])
    
    if page == "📊 Data Overview":
        show_data_overview()
    elif page == "🤖 Model Training":
        show_model_training()
    elif page == "🔮 Prediction":
        show_prediction_interface()
    elif page == "📈 Model Performance":
        show_model_performance()

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
    
    # Create distribution plots
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
    
    st.subheader("Sample Data")
    st.dataframe(df.head(10))

def show_model_training():
    st.header("🤖 Model Training")
    
    st.markdown("""
    This section allows you to train a machine learning model using gradient descent 
    to predict concrete compressive strength based on the input features.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Training Parameters")
        st.markdown("Adjust the parameters in the sidebar to configure training.")
        
        if st.button("🚀 Train Model", type="primary"):
            training_results = train_model()
            st.rerun()
    
    with col2:
        if st.session_state.model_trained:
            st.subheader("Training Results")
            results = st.session_state.training_results
            
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("Training Cost", f"{results['train_cost']:.4f}")
                st.metric("Training R²", f"{results['train_r2']:.4f}")
            with col2b:
                st.metric("Testing Cost", f"{results['test_cost']:.4f}")
                st.metric("Testing R²", f"{results['test_r2']:.4f}")
    
    if st.session_state.model_trained:
        st.subheader("Training Progress")
        
        # Plot cost history
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
        input_values['Cement'] = st.slider(
            "Cement (kg/m³)", 
            min_value=float(feature_ranges['Cement'][0]), 
            max_value=float(feature_ranges['Cement'][1]), 
            value=400.0,
            step=10.0
        )
        
        input_values['BlastFurnaceSlag'] = st.slider(
            "Blast Furnace Slag (kg/m³)", 
            min_value=float(feature_ranges['BlastFurnaceSlag'][0]), 
            max_value=float(feature_ranges['BlastFurnaceSlag'][1]), 
            value=100.0,
            step=10.0
        )
        
        input_values['FlyAsh'] = st.slider(
            "Fly Ash (kg/m³)", 
            min_value=float(feature_ranges['FlyAsh'][0]), 
            max_value=float(feature_ranges['FlyAsh'][1]), 
            value=50.0,
            step=5.0
        )
        
        input_values['Water'] = st.slider(
            "Water (kg/m³)", 
            min_value=float(feature_ranges['Water'][0]), 
            max_value=float(feature_ranges['Water'][1]), 
            value=180.0,
            step=5.0
        )
    
    with col2:
        st.subheader("Additives & Aggregates")
        input_values['Superplasticizer'] = st.slider(
            "Superplasticizer (kg/m³)", 
            min_value=float(feature_ranges['Superplasticizer'][0]), 
            max_value=float(feature_ranges['Superplasticizer'][1]), 
            value=10.0,
            step=1.0
        )
        
        input_values['CoarseAggregate'] = st.slider(
            "Coarse Aggregate (kg/m³)", 
            min_value=float(feature_ranges['CoarseAggregate'][0]), 
            max_value=float(feature_ranges['CoarseAggregate'][1]), 
            value=1000.0,
            step=50.0
        )
        
        input_values['FineAggregate'] = st.slider(
            "Fine Aggregate (kg/m³)", 
            min_value=float(feature_ranges['FineAggregate'][0]), 
            max_value=float(feature_ranges['FineAggregate'][1]), 
            value=750.0,
            step=25.0
        )
        
        input_values['Age'] = st.slider(
            "Age (days)", 
            min_value=int(feature_ranges['Age'][0]), 
            max_value=int(feature_ranges['Age'][1]), 
            value=28,
            step=1
        )
    
    # Make prediction
    input_array = np.array([
        input_values['Cement'],
        input_values['BlastFurnaceSlag'],
        input_values['FlyAsh'],
        input_values['Water'],
        input_values['Superplasticizer'],
        input_values['CoarseAggregate'],
        input_values['FineAggregate'],
        input_values['Age']
    ])
    
    prediction = st.session_state.model.predict(input_array)
    
    st.markdown("---")
    st.subheader("🎯 Prediction Result")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.metric(
            label="Predicted Concrete Strength",
            value=f"{prediction:.2f} MPa",
            delta=None
        )
        
        # Add strength classification
        if prediction < 20:
            strength_class = "Low Strength"
            color = "🔴"
        elif prediction < 40:
            strength_class = "Medium Strength"
            color = "🟡"
        elif prediction < 60:
            strength_class = "High Strength"
            color = "🟢"
        else:
            strength_class = "Very High Strength"
            color = "🟢"
        
        st.markdown(f"**Classification:** {color} {strength_class}")
    
    # Show input summary
    st.subheader("📋 Input Summary")
    input_df = pd.DataFrame([input_values])
    st.dataframe(input_df, use_container_width=True)
    
    # Preset examples
    st.subheader("🔧 Quick Examples")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("High Strength Mix"):
            st.session_state.example_values = {
                'Cement': 500, 'BlastFurnaceSlag': 150, 'FlyAsh': 0,
                'Water': 150, 'Superplasticizer': 20, 'CoarseAggregate': 1100,
                'FineAggregate': 800, 'Age': 28
            }
            st.rerun()
    
    with col2:
        if st.button("Standard Mix"):
            st.session_state.example_values = {
                'Cement': 350, 'BlastFurnaceSlag': 100, 'FlyAsh': 50,
                'Water': 180, 'Superplasticizer': 10, 'CoarseAggregate': 1000,
                'FineAggregate': 750, 'Age': 28
            }
            st.rerun()
    
    with col3:
        if st.button("Eco-Friendly Mix"):
            st.session_state.example_values = {
                'Cement': 250, 'BlastFurnaceSlag': 200, 'FlyAsh': 100,
                'Water': 170, 'Superplasticizer': 15, 'CoarseAggregate': 950,
                'FineAggregate': 700, 'Age': 28
            }
            st.rerun()

def show_model_performance():
    st.header("📈 Model Performance")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first in the 'Model Training' section.")
        return
    
    results = st.session_state.training_results
    
    st.subheader("Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Training R²", f"{results['train_r2']:.4f}")
    with col2:
        st.metric("Testing R²", f"{results['test_r2']:.4f}")
    with col3:
        st.metric("Training Cost", f"{results['train_cost']:.4f}")
    with col4:
        st.metric("Testing Cost", f"{results['test_cost']:.4f}")
    
    # Prediction vs Actual plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Training Set: Predicted vs Actual")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results['y_train'],
            y=results['train_predictions'],
            mode='markers',
            name='Predictions',
            opacity=0.6
        ))
        
        # Add perfect prediction line
        min_val = min(results['y_train'].min(), results['train_predictions'].min())
        max_val = max(results['y_train'].max(), results['train_predictions'].max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            xaxis_title="Actual Strength (MPa)",
            yaxis_title="Predicted Strength (MPa)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Test Set: Predicted vs Actual")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results['y_test'],
            y=results['test_predictions'],
            mode='markers',
            name='Predictions',
            opacity=0.6
        ))
        
        # Add perfect prediction line
        min_val = min(results['y_test'].min(), results['test_predictions'].min())
        max_val = max(results['y_test'].max(), results['test_predictions'].max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            xaxis_title="Actual Strength (MPa)",
            yaxis_title="Predicted Strength (MPa)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance (weights visualization)
    st.subheader("Feature Importance (Model Weights)")
    
    feature_names = list(get_feature_names().keys())
    weights = st.session_state.model.weights
    
    fig = go.Figure(data=[
        go.Bar(x=feature_names, y=weights, text=np.round(weights, 3), textposition='auto')
    ])
    fig.update_layout(
        title="Model Weights for Each Feature",
        xaxis_title="Features",
        yaxis_title="Weight Value",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Cost history
    st.subheader("Training Cost History")
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
    
    # Model save/load functionality
    st.subheader("Model Management")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Save Model"):
            st.session_state.model.save_model('concrete_strength_model.npz')
            st.success("Model saved successfully!")
    
    with col2:
        if st.button("📁 Load Model"):
            if st.session_state.model.load_model('concrete_strength_model.npz'):
                st.success("Model loaded successfully!")
                st.session_state.model_trained = True
            else:
                st.error("No saved model found!")

if __name__ == "__main__":
    main()
