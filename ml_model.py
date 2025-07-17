import numpy as np
import pandas as pd
from utilities import normalize_features, eval_cost, eval_gradient, predict_strength
from sklearn.model_selection import train_test_split

class ConcreteStrengthModel:
    def __init__(self):
        self.weights = None
        self.bias = None
        self.feature_mean = None
        self.feature_std = None
        self.cost_history = []
        self.is_trained = False
        
    def prepare_data(self, df):
        """
        Prepare data for training
        """
        # Extract features and target
        feature_columns = ['Cement', 'BlastFurnaceSlag', 'FlyAsh', 'Water', 
                          'Superplasticizer', 'CoarseAggregate', 'FineAggregate', 'Age']
        
        X = df[feature_columns].values
        y = df['ConcreteStrength'].values
        
        return X, y
    
    def train(self, X, y, learning_rate=0.01, num_iterations=1000, test_size=0.2, random_state=42):
        """
        Train the concrete strength prediction model
        """
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
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
            # Calculate cost
            cost = eval_cost(X_train_norm, y_train, self.weights, self.bias)
            self.cost_history.append(cost)
            
            # Calculate gradients
            dJdw, dJdb = eval_gradient(X_train_norm, y_train, self.weights, self.bias)
            
            # Update parameters
            self.weights = self.weights - learning_rate * dJdw
            self.bias = self.bias - learning_rate * dJdb
        
        self.is_trained = True
        
        # Calculate final training and testing costs
        train_cost = eval_cost(X_train_norm, y_train, self.weights, self.bias)
        test_cost = eval_cost(X_test_norm, y_test, self.weights, self.bias)
        
        # Calculate R² score for training and testing sets
        train_predictions = self.predict_batch(X_train)
        test_predictions = self.predict_batch(X_test)
        
        train_r2 = self.calculate_r2(y_train, train_predictions)
        test_r2 = self.calculate_r2(y_test, test_predictions)
        
        return {
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
    
    def predict(self, input_features):
        """
        Predict concrete strength for a single sample
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return predict_strength(
            input_features, self.weights, self.bias, 
            self.feature_mean, self.feature_std
        )
    
    def predict_batch(self, X):
        """
        Predict concrete strength for multiple samples
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = []
        for i in range(X.shape[0]):
            pred = self.predict(X[i])
            predictions.append(pred)
        
        return np.array(predictions)
    
    def calculate_r2(self, y_true, y_pred):
        """
        Calculate R² score
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2
    
    def save_model(self, filename='concrete_strength_model.npz'):
        """
        Save trained model parameters
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        guidelines = """
x_norm = (x_input - feature_mean) / feature_std   # This is a numpy broadcasting operation
y_predict = np.dot(x_norm, weights_norm) + bias_norm

feature_title = ['Cement Quantity',      # Kg/m3
                 'Blast Furnace Slag',   # Kg/m3
                 'Fly Ash',              # Kg/m3
                 'Water',                # Kg/m3
                 'Superplasticizer',     # Kg/m3
                 'Coarse Aggregate',     # Kg/m3
                 'Fine Aggregate',       # Kg/m3
                 'Age']                  # days
                 
Only one output is expected, that is concrete strength in MPa
"""
        
        np.savez(filename,
                weights_norm=self.weights,
                bias_norm=self.bias,
                feature_mean=self.feature_mean,
                feature_std=self.feature_std,
                pred_guidelines=guidelines)
    
    def load_model(self, filename='concrete_strength_model.npz'):
        """
        Load trained model parameters
        """
        try:
            model_data = np.load(filename)
            self.weights = model_data['weights_norm']
            self.bias = model_data['bias_norm']
            self.feature_mean = model_data['feature_mean']
            self.feature_std = model_data['feature_std']
            self.is_trained = True
            return True
        except FileNotFoundError:
            return False
