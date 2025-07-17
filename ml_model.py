import numpy as np
import pandas as pd
from utilities import normalize_features, eval_cost, eval_gradient, predict_strength
from sklearn.model_selection import train_test_split
from database import DatabaseManager

class ConcreteStrengthModel:
    def __init__(self):
        self.weights = None
        self.bias = None
        self.feature_mean = None
        self.feature_std = None
        self.cost_history = []
        self.is_trained = False
        self.model_id = None
        self.model_name = None
        
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
        # Store training parameters for database saving
        self.last_learning_rate = learning_rate
        self.last_num_iterations = num_iterations
        self.last_test_size = test_size
        
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
        
        # Store results for database saving
        results = {
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
        self.last_training_results = results
        
        return results
    
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
    
    def save_model(self, model_name='default_model', filename='concrete_strength_model.npz'):
        """
        Save trained model parameters to database and file
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Save to database
        db_manager = DatabaseManager()
        try:
            training_params = {
                'learning_rate': getattr(self, 'last_learning_rate', 0.01),
                'num_iterations': getattr(self, 'last_num_iterations', 1000),
                'test_size': getattr(self, 'last_test_size', 0.2)
            }
            
            training_results = getattr(self, 'last_training_results', {})
            
            self.model_id = db_manager.save_model(
                self, model_name, training_params, training_results
            )
            self.model_name = model_name
            
            if self.model_id:
                print(f"Model saved to database with ID: {self.model_id}")
        except Exception as e:
            print(f"Failed to save model to database: {e}")
        finally:
            db_manager.close()
        
        # Save to file (backup)
        try:
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
        except Exception as e:
            print(f"Failed to save model to file: {e}")
    
    def load_model(self, model_name=None, filename='concrete_strength_model.npz'):
        """
        Load trained model parameters from database or file
        """
        # Try to load from database first
        db_manager = DatabaseManager()
        try:
            model_data = db_manager.load_model(model_name)
            if model_data:
                self.weights = model_data['weights']
                self.bias = model_data['bias']
                self.feature_mean = model_data['feature_mean']
                self.feature_std = model_data['feature_std']
                self.model_id = model_data['id']
                self.model_name = model_data['model_name']
                self.is_trained = True
                print(f"Model '{model_data['model_name']}' loaded from database")
                return True
        except Exception as e:
            print(f"Failed to load model from database: {e}")
        finally:
            db_manager.close()
        
        # Fallback to file loading
        try:
            model_data = np.load(filename)
            self.weights = model_data['weights_norm']
            self.bias = model_data['bias_norm']
            self.feature_mean = model_data['feature_mean']
            self.feature_std = model_data['feature_std']
            self.is_trained = True
            print(f"Model loaded from file: {filename}")
            return True
        except FileNotFoundError:
            print(f"No model file found: {filename}")
            return False
    
    def predict_and_save(self, input_features, notes=None):
        """
        Make prediction and save to database
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Make prediction
        prediction = self.predict(input_features)
        
        # Classify strength
        if prediction < 20:
            classification = "Low Strength"
        elif prediction < 40:
            classification = "Medium Strength"
        elif prediction < 60:
            classification = "High Strength"
        else:
            classification = "Very High Strength"
        
        # Save to database if model_id is available
        if self.model_id:
            db_manager = DatabaseManager()
            try:
                db_manager.save_prediction(
                    self.model_id, input_features, prediction, classification, notes
                )
            except Exception as e:
                print(f"Failed to save prediction to database: {e}")
            finally:
                db_manager.close()
        
        return prediction, classification
