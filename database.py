import os
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import ARRAY
import numpy as np

# Database setup
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost/concrete_db')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ConcreteDataset(Base):
    """Table to store concrete dataset records"""
    __tablename__ = "concrete_datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    cement = Column(Float, nullable=False)
    blast_furnace_slag = Column(Float, nullable=False)
    fly_ash = Column(Float, nullable=False)
    water = Column(Float, nullable=False)
    superplasticizer = Column(Float, nullable=False)
    coarse_aggregate = Column(Float, nullable=False)
    fine_aggregate = Column(Float, nullable=False)
    age = Column(Integer, nullable=False)
    concrete_strength = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    source = Column(String(50), default='synthetic')  # 'synthetic', 'uploaded', 'manual'

class TrainedModel(Base):
    """Table to store trained model parameters"""
    __tablename__ = "trained_models"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False)
    weights = Column(Text, nullable=False)  # JSON string of weights array
    bias = Column(Float, nullable=False)
    feature_mean = Column(Text, nullable=False)  # JSON string of feature means
    feature_std = Column(Text, nullable=False)  # JSON string of feature stds
    learning_rate = Column(Float, nullable=False)
    num_iterations = Column(Integer, nullable=False)
    test_size = Column(Float, nullable=False)
    train_r2 = Column(Float, nullable=True)
    test_r2 = Column(Float, nullable=True)
    train_cost = Column(Float, nullable=True)
    test_cost = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class PredictionHistory(Base):
    """Table to store prediction history"""
    __tablename__ = "prediction_history"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, nullable=False)  # Reference to trained model
    cement = Column(Float, nullable=False)
    blast_furnace_slag = Column(Float, nullable=False)
    fly_ash = Column(Float, nullable=False)
    water = Column(Float, nullable=False)
    superplasticizer = Column(Float, nullable=False)
    coarse_aggregate = Column(Float, nullable=False)
    fine_aggregate = Column(Float, nullable=False)
    age = Column(Integer, nullable=False)
    predicted_strength = Column(Float, nullable=False)
    strength_classification = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text, nullable=True)

class TrainingSession(Base):
    """Table to store training session details"""
    __tablename__ = "training_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, nullable=False)
    dataset_size = Column(Integer, nullable=False)
    training_time_seconds = Column(Float, nullable=True)
    cost_history = Column(Text, nullable=True)  # JSON string of cost history
    final_weights = Column(Text, nullable=False)  # JSON string of final weights
    convergence_achieved = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class DatabaseManager:
    """Database operations manager"""
    
    def __init__(self):
        self.session = SessionLocal()
    
    def close(self):
        """Close database session"""
        self.session.close()
    
    def save_dataset(self, df, source='synthetic'):
        """Save concrete dataset to database"""
        try:
            # Clear existing synthetic data if adding new synthetic data
            if source == 'synthetic':
                self.session.query(ConcreteDataset).filter(
                    ConcreteDataset.source == 'synthetic'
                ).delete()
            
            # Add new records
            for _, row in df.iterrows():
                record = ConcreteDataset(
                    cement=float(row['Cement']),
                    blast_furnace_slag=float(row['BlastFurnaceSlag']),
                    fly_ash=float(row['FlyAsh']),
                    water=float(row['Water']),
                    superplasticizer=float(row['Superplasticizer']),
                    coarse_aggregate=float(row['CoarseAggregate']),
                    fine_aggregate=float(row['FineAggregate']),
                    age=int(row['Age']),
                    concrete_strength=float(row['ConcreteStrength']),
                    source=source
                )
                self.session.add(record)
            
            self.session.commit()
            return True
        except Exception as e:
            self.session.rollback()
            print(f"Error saving dataset: {e}")
            return False
    
    def load_dataset(self):
        """Load concrete dataset from database"""
        try:
            records = self.session.query(ConcreteDataset).all()
            
            if not records:
                return None
            
            data = []
            for record in records:
                data.append({
                    'Cement': record.cement,
                    'BlastFurnaceSlag': record.blast_furnace_slag,
                    'FlyAsh': record.fly_ash,
                    'Water': record.water,
                    'Superplasticizer': record.superplasticizer,
                    'CoarseAggregate': record.coarse_aggregate,
                    'FineAggregate': record.fine_aggregate,
                    'Age': record.age,
                    'ConcreteStrength': record.concrete_strength
                })
            
            import pandas as pd
            return pd.DataFrame(data)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def save_model(self, model, model_name, training_params, training_results):
        """Save trained model to database"""
        try:
            # Deactivate previous models with same name
            self.session.query(TrainedModel).filter(
                TrainedModel.model_name == model_name
            ).update({TrainedModel.is_active: False})
            
            # Save new model
            model_record = TrainedModel(
                model_name=model_name,
                weights=json.dumps(model.weights.tolist()),
                bias=float(model.bias),
                feature_mean=json.dumps(model.feature_mean.tolist()),
                feature_std=json.dumps(model.feature_std.tolist()),
                learning_rate=training_params['learning_rate'],
                num_iterations=training_params['num_iterations'],
                test_size=training_params['test_size'],
                train_r2=training_results.get('train_r2'),
                test_r2=training_results.get('test_r2'),
                train_cost=training_results.get('train_cost'),
                test_cost=training_results.get('test_cost'),
                is_active=True
            )
            self.session.add(model_record)
            self.session.commit()
            
            return model_record.id
        except Exception as e:
            self.session.rollback()
            print(f"Error saving model: {e}")
            return None
    
    def load_model(self, model_name=None):
        """Load trained model from database"""
        try:
            if model_name:
                record = self.session.query(TrainedModel).filter(
                    TrainedModel.model_name == model_name,
                    TrainedModel.is_active == True
                ).first()
            else:
                record = self.session.query(TrainedModel).filter(
                    TrainedModel.is_active == True
                ).order_by(TrainedModel.created_at.desc()).first()
            
            if not record:
                return None
            
            # Return model parameters
            return {
                'id': record.id,
                'model_name': record.model_name,
                'weights': np.array(json.loads(record.weights)),
                'bias': record.bias,
                'feature_mean': np.array(json.loads(record.feature_mean)),
                'feature_std': np.array(json.loads(record.feature_std)),
                'train_r2': record.train_r2,
                'test_r2': record.test_r2,
                'train_cost': record.train_cost,
                'test_cost': record.test_cost,
                'created_at': record.created_at
            }
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def save_prediction(self, model_id, inputs, prediction, classification, notes=None):
        """Save prediction to history"""
        try:
            prediction_record = PredictionHistory(
                model_id=model_id,
                cement=float(inputs[0]),
                blast_furnace_slag=float(inputs[1]),
                fly_ash=float(inputs[2]),
                water=float(inputs[3]),
                superplasticizer=float(inputs[4]),
                coarse_aggregate=float(inputs[5]),
                fine_aggregate=float(inputs[6]),
                age=int(inputs[7]),
                predicted_strength=float(prediction),
                strength_classification=classification,
                notes=notes
            )
            self.session.add(prediction_record)
            self.session.commit()
            return True
        except Exception as e:
            self.session.rollback()
            print(f"Error saving prediction: {e}")
            return False
    
    def get_prediction_history(self, limit=50):
        """Get recent prediction history"""
        try:
            records = self.session.query(PredictionHistory).order_by(
                PredictionHistory.created_at.desc()
            ).limit(limit).all()
            
            history = []
            for record in records:
                history.append({
                    'id': record.id,
                    'cement': record.cement,
                    'blast_furnace_slag': record.blast_furnace_slag,
                    'fly_ash': record.fly_ash,
                    'water': record.water,
                    'superplasticizer': record.superplasticizer,
                    'coarse_aggregate': record.coarse_aggregate,
                    'fine_aggregate': record.fine_aggregate,
                    'age': record.age,
                    'predicted_strength': record.predicted_strength,
                    'strength_classification': record.strength_classification,
                    'created_at': record.created_at,
                    'notes': record.notes
                })
            
            return history
        except Exception as e:
            print(f"Error getting prediction history: {e}")
            return []
    
    def get_model_list(self):
        """Get list of all trained models"""
        try:
            records = self.session.query(TrainedModel).order_by(
                TrainedModel.created_at.desc()
            ).all()
            
            models = []
            for record in records:
                models.append({
                    'id': record.id,
                    'model_name': record.model_name,
                    'train_r2': record.train_r2,
                    'test_r2': record.test_r2,
                    'created_at': record.created_at,
                    'is_active': record.is_active
                })
            
            return models
        except Exception as e:
            print(f"Error getting model list: {e}")
            return []
    
    def get_dataset_stats(self):
        """Get dataset statistics"""
        try:
            total_records = self.session.query(ConcreteDataset).count()
            
            if total_records == 0:
                return None
            
            # Get basic stats
            from sqlalchemy import func
            stats = self.session.query(
                func.min(ConcreteDataset.concrete_strength).label('min_strength'),
                func.max(ConcreteDataset.concrete_strength).label('max_strength'),
                func.avg(ConcreteDataset.concrete_strength).label('avg_strength'),
                func.count(ConcreteDataset.id).label('total_count')
            ).first()
            
            return {
                'total_records': total_records,
                'min_strength': float(stats.min_strength),
                'max_strength': float(stats.max_strength),
                'avg_strength': float(stats.avg_strength),
                'total_count': stats.total_count
            }
        except Exception as e:
            print(f"Error getting dataset stats: {e}")
            return None

# Initialize database
def init_database():
    """Initialize database with tables"""
    try:
        create_tables()
        print("Database tables created successfully")
        return True
    except Exception as e:
        print(f"Error initializing database: {e}")
        return False

if __name__ == "__main__":
    init_database()