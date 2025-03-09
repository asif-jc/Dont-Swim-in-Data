# src/models/benchmarks/lightgbm_models.py
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sys
from sklearn.base import BaseEstimator, RegressorMixin

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

from src.utils.logging import setup_logger
from src.data.preprocessing import Preprocessor


logger = setup_logger(__name__)

class LightGBMModel(BaseEstimator, RegressorMixin):
    
    def __init__(self, 
                 config: Dict[str, Any], 
                 random_state: Optional[int] = None):
        """Initialize the LightGBM model with Poisson loss.
        
        Args:
            config: Dictionary containing model hyperparameters
            random_state: Random seed for reproducibility
        """
        self.config = config
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        
        # Default hyperparameters
        self.n_estimators = config.get("n_estimators", 100)
        self.learning_rate = config.get("learning_rate", 0.05)
        self.max_depth = config.get("max_depth", -1)  # -1 means no limit
        self.num_leaves = config.get("num_leaves", 31)
        self.min_data_in_leaf = config.get("min_data_in_leaf", 20)
        self.bagging_fraction = config.get("bagging_fraction", 0.8)
        self.bagging_freq = config.get("bagging_freq", 5)
        self.feature_fraction = config.get("feature_fraction", 0.8)
        self.lambda_l1 = config.get("lambda_l1", 0.0)
        self.lambda_l2 = config.get("lambda_l2", 0.0)
        
        logger.info(f"Initialized LightGBM model with Poisson loss")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LightGBMModel':
        """Train the LightGBM model on the provided data.
        
        Args:
            X: Features DataFrame
            y: Target variable Series
            
        Returns:
            Self reference for method chaining
        """
        self.feature_names = X.columns.tolist()

        X["DateTime"] = 0
        
        # Create LightGBM dataset
        train_data = lgb.Dataset(
            X, 
            label=y,
            feature_name=self.feature_names
        )
        
        # Set model parameters with Poisson objective
        params = {
            'objective': 'poisson',
            'metric': 'poisson',
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'num_leaves': self.num_leaves,
            'min_data_in_leaf': self.min_data_in_leaf,
            'bagging_fraction': self.bagging_fraction,
            'bagging_freq': self.bagging_freq,
            'feature_fraction': self.feature_fraction,
            'lambda_l1': self.lambda_l1,
            'lambda_l2': self.lambda_l2,
            'verbose': -1,
            'seed': self.random_state
        }
        
        # Train model
        logger.info(f"Training LightGBM model with {len(X)} samples")
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.n_estimators,
        )
        
        logger.info(f"Model training completed with {self.model.num_trees()} trees")
        return self
    
    def apply_enterococci_constraints(self, predictions):
        # Apply minimum value of 5 MPN/100mL
        constrained = np.maximum(predictions, 5)
        # Round to nearest integer
        constrained = np.round(constrained)
        return constrained
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions using the trained model.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Array of predicted values
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() before predict().")
        
        logger.info(f"Generating predictions for {len(X)} samples")
        X["DateTime"] = 0

        predictions = self.model.predict(X)
        predictions = self.apply_enterococci_constraints(predictions)

        predictions = pd.Series(predictions, name="predictions")

        return predictions
    

    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() before get_feature_importance().")
        
        importance = self.model.feature_importance(importance_type='gain')
        return dict(zip(self.feature_names, importance))
    
    def save(self, path: Union[str, Path]) -> None:
        """Save the trained model to disk.
        
        Args:
            path: Path where to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() before save().")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the entire object for consistency with other models
        joblib.dump(self, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'LightGBMModel':
        """Load a trained model from disk.
        
        Args:
            path: Path from where to load the model
            
        Returns:
            Loaded model
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")
        
        model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return model

    def train(self, data: pd.DataFrame) -> 'LightGBMModel':
        """Higher-level training method for the pipeline.
        
        This method handles data splitting and preprocessing before training.
        
        Args:
            data: DataFrame containing features and target
            
        Returns:
            Self reference for method chaining
        """
        # Extract features and target
        X = data.drop('Enterococci', axis=1)  # Adjust column name if needed
        y = data['Enterococci']  # Adjust column name if needed
        
        # Train the model
        return self.fit(X, y)