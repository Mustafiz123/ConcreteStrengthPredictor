"""
Flask Web Application for Concrete Strength Prediction
Simple, clean interface with core prediction functionality
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import json
import os
from datetime import datetime

# Import ML utilities
from utilities import normalize_features, eval_cost, eval_gradient, predict_strength

app = Flask(__name__)

class ConcreteModel:
    """Simple concrete strength prediction model"""
    
    def __init__(self):
        self.weights = None
        self.bias = None
        self.feature_mean = None
        self.feature_std = None
        self.is_trained = False
        self.model_info = {}
    
    def create_sample_data(self, n_samples=1000):
        """Generate sample concrete dataset for training"""
        np.random.seed(42)
        
        # Generate realistic concrete mix data
        data = {
            'cement': np.random.uniform(100, 600, n_samples),
            'blast_furnace_slag': np.random.uniform(0, 400, n_samples),
            'fly_ash': np.random.uniform(0, 250, n_samples),
            'water': np.random.uniform(120, 250, n_samples),
            'superplasticizer': np.random.uniform(0, 35, n_samples),
            'coarse_aggregate': np.random.uniform(700, 1200, n_samples),
            'fine_aggregate': np.random.uniform(500, 950, n_samples),
            'age': np.random.randint(1, 365, n_samples)
        }
        
        # Create realistic strength values based on concrete engineering principles
        strength = (
            data['cement'] * 0.1 +
            data['blast_furnace_slag'] * 0.08 +
            data['fly_ash'] * 0.06 +
            data['water'] * (-0.15) +
            data['superplasticizer'] * 0.5 +
            data['coarse_aggregate'] * 0.01 +
            data['fine_aggregate'] * 0.01 +
            np.log(data['age'] + 1) * 5 +
            np.random.normal(0, 5, n_samples)
        )
        
        # Clip to realistic concrete strength range
        strength = np.clip(strength, 10, 100)
        
        # Prepare features matrix
        X = np.column_stack([
            data['cement'], data['blast_furnace_slag'], data['fly_ash'],
            data['water'], data['superplasticizer'], data['coarse_aggregate'],
            data['fine_aggregate'], data['age']
        ])
        
        return X, strength
    
    def train_model(self, learning_rate=0.01, num_iterations=1000):
        """Train the concrete strength prediction model"""
        # Generate training data
        X, y = self.create_sample_data()
        
        # Normalize features
        X_norm, self.feature_mean, self.feature_std = normalize_features(X)
        
        # Initialize parameters
        n_features = X_norm.shape[1]
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0.0
        
        # Training loop with gradient descent
        cost_history = []
        for i in range(num_iterations):
            cost = eval_cost(X_norm, y, self.weights, self.bias)
            cost_history.append(cost)
            
            # Compute gradients
            dJdw, dJdb = eval_gradient(X_norm, y, self.weights, self.bias)
            
            # Update parameters
            self.weights = self.weights - learning_rate * dJdw
            self.bias = self.bias - learning_rate * dJdb
        
        self.is_trained = True
        
        # Calculate final training metrics
        final_cost = cost_history[-1]
        initial_cost = cost_history[0]
        
        self.model_info = {
            'training_samples': len(X),
            'final_cost': final_cost,
            'initial_cost': initial_cost,
            'cost_reduction': ((initial_cost - final_cost) / initial_cost) * 100,
            'learning_rate': learning_rate,
            'iterations': num_iterations,
            'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return True
    
    def predict(self, cement, blast_furnace_slag, fly_ash, water, superplasticizer, 
                coarse_aggregate, fine_aggregate, age):
        """Make concrete strength prediction"""
        if not self.is_trained:
            # Auto-train if not trained
            self.train_model()
        
        # Prepare input array
        input_features = np.array([
            cement, blast_furnace_slag, fly_ash, water,
            superplasticizer, coarse_aggregate, fine_aggregate, age
        ])
        
        # Make prediction
        prediction = predict_strength(
            input_features, self.weights, self.bias,
            self.feature_mean, self.feature_std
        )
        
        return prediction
    
    def classify_strength(self, strength):
        """Classify concrete strength level"""
        if strength < 20:
            return "Low Strength", "danger"
        elif strength < 40:
            return "Medium Strength", "warning"
        elif strength < 60:
            return "High Strength", "success"
        else:
            return "Very High Strength", "primary"

# Initialize global model
concrete_model = ConcreteModel()

@app.route('/')
def index():
    """Main prediction interface"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get input values from form
        cement = float(request.form['cement'])
        blast_furnace_slag = float(request.form['blast_furnace_slag'])
        fly_ash = float(request.form['fly_ash'])
        water = float(request.form['water'])
        superplasticizer = float(request.form['superplasticizer'])
        coarse_aggregate = float(request.form['coarse_aggregate'])
        fine_aggregate = float(request.form['fine_aggregate'])
        age = int(request.form['age'])
        
        # Validate inputs
        if cement < 0 or water < 0 or age < 1:
            return jsonify({'error': 'Invalid input values'}), 400
        
        # Make prediction
        predicted_strength = concrete_model.predict(
            cement, blast_furnace_slag, fly_ash, water,
            superplasticizer, coarse_aggregate, fine_aggregate, age
        )
        
        # Classify strength
        classification, badge_type = concrete_model.classify_strength(predicted_strength)
        
        # Prepare response
        result = {
            'predicted_strength': round(predicted_strength, 2),
            'classification': classification,
            'badge_type': badge_type,
            'model_info': concrete_model.model_info,
            'inputs': {
                'cement': cement,
                'blast_furnace_slag': blast_furnace_slag,
                'fly_ash': fly_ash,
                'water': water,
                'superplasticizer': superplasticizer,
                'coarse_aggregate': coarse_aggregate,
                'fine_aggregate': fine_aggregate,
                'age': age
            }
        }
        
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({'error': 'Invalid input format'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model-info')
def model_info():
    """Get model training information"""
    if not concrete_model.is_trained:
        concrete_model.train_model()
    
    return jsonify(concrete_model.model_info)

@app.route('/retrain', methods=['POST'])
def retrain_model():
    """Retrain the model with new parameters"""
    try:
        learning_rate = float(request.form.get('learning_rate', 0.01))
        iterations = int(request.form.get('iterations', 1000))
        
        # Validate parameters
        if learning_rate <= 0 or learning_rate > 1:
            return jsonify({'error': 'Learning rate must be between 0 and 1'}), 400
        
        if iterations < 100 or iterations > 10000:
            return jsonify({'error': 'Iterations must be between 100 and 10000'}), 400
        
        # Retrain model
        success = concrete_model.train_model(learning_rate, iterations)
        
        if success:
            return jsonify({
                'message': 'Model retrained successfully',
                'model_info': concrete_model.model_info
            })
        else:
            return jsonify({'error': 'Training failed'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Train model on startup
    print("Training concrete strength prediction model...")
    concrete_model.train_model()
    print("Model ready!")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)