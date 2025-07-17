import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_loader import load_concrete_data, get_feature_names, get_feature_ranges
from ml_model import ConcreteStrengthModel
from database import DatabaseManager, init_database

# Configure page
st.set_page_config(
    page_title="Concrete Strength Prediction",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
init_database()

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = ConcreteStrengthModel()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = DatabaseManager()

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
        "📈 Model Performance",
        "💾 Database Management",
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
    elif page == "💾 Database Management":
        show_database_management()
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
    
    # Use predict_and_save to automatically save to database
    try:
        prediction, classification = st.session_state.model.predict_and_save(input_array)
    except:
        # Fallback to regular prediction if database save fails
        prediction = st.session_state.model.predict(input_array)
        # Classify strength
        if prediction < 20:
            classification = "Low Strength"
        elif prediction < 40:
            classification = "Medium Strength"
        elif prediction < 60:
            classification = "High Strength"
        else:
            classification = "Very High Strength"
    
    st.markdown("---")
    st.subheader("🎯 Prediction Result")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.metric(
            label="Predicted Concrete Strength",
            value=f"{prediction:.2f} MPa",
            delta=None
        )
        
        # Display classification with color indicator
        if classification == "Low Strength":
            color = "🔴"
        elif classification == "Medium Strength":
            color = "🟡"
        elif classification == "High Strength":
            color = "🟢"
        else:
            color = "🟢"
        
        st.markdown(f"**Classification:** {color} {classification}")
    
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
            if st.session_state.model.load_model():
                st.success("Model loaded successfully!")
                st.session_state.model_trained = True
            else:
                st.error("No saved model found!")

def show_database_management():
    st.header("💾 Database Management")
    
    # Database statistics
    st.subheader("Database Statistics")
    
    db_manager = DatabaseManager()
    try:
        # Dataset stats
        dataset_stats = db_manager.get_dataset_stats()
        if dataset_stats:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", dataset_stats['total_count'])
            with col2:
                st.metric("Min Strength", f"{dataset_stats['min_strength']:.1f} MPa")
            with col3:
                st.metric("Max Strength", f"{dataset_stats['max_strength']:.1f} MPa")
            with col4:
                st.metric("Avg Strength", f"{dataset_stats['avg_strength']:.1f} MPa")
        else:
            st.info("No dataset records found in database")
        
        # Model management
        st.subheader("Saved Models")
        models = db_manager.get_model_list()
        
        if models:
            # Create a DataFrame for better display
            model_df = pd.DataFrame(models)
            model_df['created_at'] = pd.to_datetime(model_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
            
            # Display models table
            st.dataframe(
                model_df[['model_name', 'train_r2', 'test_r2', 'created_at', 'is_active']],
                use_container_width=True
            )
            
            # Model selection and loading
            col1, col2 = st.columns(2)
            with col1:
                selected_model = st.selectbox(
                    "Select Model to Load",
                    options=[m['model_name'] for m in models],
                    index=0
                )
            
            with col2:
                if st.button("Load Selected Model"):
                    if st.session_state.model.load_model(selected_model):
                        st.success(f"Model '{selected_model}' loaded successfully!")
                        st.session_state.model_trained = True
                        st.rerun()
                    else:
                        st.error("Failed to load model")
        else:
            st.info("No trained models found in database")
        
        # Model saving
        st.subheader("Save Current Model")
        if st.session_state.model_trained:
            model_name = st.text_input("Model Name", value="my_concrete_model")
            if st.button("Save Model to Database"):
                st.session_state.model.save_model(model_name)
                st.success("Model saved to database!")
                st.rerun()
        else:
            st.warning("Train a model first before saving")
    
    except Exception as e:
        st.error(f"Database error: {str(e)}")
    finally:
        db_manager.close()

def show_prediction_history():
    st.header("📋 Prediction History")
    
    db_manager = DatabaseManager()
    try:
        # Get prediction history
        history = db_manager.get_prediction_history(limit=100)
        
        if history:
            st.subheader(f"Recent Predictions ({len(history)} records)")
            
            # Convert to DataFrame for better display
            history_df = pd.DataFrame(history)
            history_df['created_at'] = pd.to_datetime(history_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
            
            # Display summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Predictions", len(history))
            with col2:
                avg_strength = history_df['predicted_strength'].mean()
                st.metric("Avg Predicted Strength", f"{avg_strength:.1f} MPa")
            with col3:
                max_strength = history_df['predicted_strength'].max()
                st.metric("Max Predicted Strength", f"{max_strength:.1f} MPa")
            with col4:
                # Count by classification
                most_common = history_df['strength_classification'].mode().iloc[0]
                st.metric("Most Common Class", most_common)
            
            # Classification distribution
            st.subheader("Strength Classification Distribution")
            classification_counts = history_df['strength_classification'].value_counts()
            
            fig = go.Figure(data=[
                go.Bar(x=classification_counts.index, y=classification_counts.values)
            ])
            fig.update_layout(
                title="Distribution of Strength Classifications",
                xaxis_title="Classification",
                yaxis_title="Count",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Strength prediction timeline
            st.subheader("Prediction Timeline")
            fig = go.Figure()
            
            # Convert datetime for plotting
            history_df['datetime'] = pd.to_datetime(history_df['created_at'])
            
            fig.add_trace(go.Scatter(
                x=history_df['datetime'],
                y=history_df['predicted_strength'],
                mode='markers+lines',
                name='Predicted Strength',
                hovertemplate='<b>%{y:.1f} MPa</b><br>%{x}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Predicted Strength Over Time",
                xaxis_title="Date",
                yaxis_title="Predicted Strength (MPa)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed history table
            st.subheader("Detailed History")
            
            # Select columns to display
            display_columns = [
                'cement', 'blast_furnace_slag', 'fly_ash', 'water',
                'superplasticizer', 'coarse_aggregate', 'fine_aggregate', 'age',
                'predicted_strength', 'strength_classification', 'created_at'
            ]
            
            # Format column names for display
            formatted_df = history_df[display_columns].copy()
            formatted_df.columns = [
                'Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water',
                'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age',
                'Predicted Strength', 'Classification', 'Date'
            ]
            
            # Round numerical columns
            numeric_columns = ['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water',
                             'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Predicted Strength']
            for col in numeric_columns:
                formatted_df[col] = formatted_df[col].round(1)
            
            st.dataframe(formatted_df, use_container_width=True)
            
            # Export functionality
            if st.button("📥 Download Prediction History"):
                csv = formatted_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"prediction_history_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("No prediction history found. Make some predictions first!")
    
    except Exception as e:
        st.error(f"Error loading prediction history: {str(e)}")
    finally:
        db_manager.close()

if __name__ == "__main__":
    main()
