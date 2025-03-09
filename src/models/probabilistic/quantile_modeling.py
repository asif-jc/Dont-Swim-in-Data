import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import sys

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

class ProbabilisticQuantileEnsembleModel:
    """
    Probabilistic Quantile Ensemble Model that trains an ensemble of LightGBM quantile regression models.
    Each model is trained for a specific quantile (e.g., 0.05, 0.30, 0.50, ...).
    
    This class adheres to the same interface as your existing baseline models, so it integrates seamlessly
    with the main pipeline.
    """
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config

        # Get the target column and (optionally) feature columns from the configuration.
        # self.target_column = config["data"].get("target_column", "Enterococci")
        # self.feature_columns = config["data"].get("feature_columns", None)
        self.target_column = None
        self.feature_columns = None
        
        # Retrieve quantile levels and LightGBM parameters specific to the probabilistic model.
        self.quantile_levels: List[float] = config.get(
            "quantiles", [0.05, 0.30, 0.50, 0.70, 0.80, 0.90, 0.92, 0.95, 0.98]
        )
        self.params: Dict[str, Any] = config.get("lgb_params", {
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_child_samples': 20,
            'n_estimators': 100,
            'min_data_in_leaf': 10,
            'lambda_l2': 0.45
        })
        # Dictionary to hold the trained LightGBM models keyed by quantile level.
        self.models: Dict[float, lgb.LGBMRegressor] = {}


    def train(self, data: pd.DataFrame) -> None:
        """
        Train an ensemble of quantile regression models.
        
        Args:
            data: The training data as a DataFrame.
                  It must include the target column and feature columns.
        """
        self.logger.info("Starting training for Probabilistic Quantile Ensemble Model.")

        eval_results = {}
        
        # Determine feature columns if not explicitly provided.
        if self.feature_columns is None:
            self.feature_columns = [col for col in data.columns if col != self.target_column]
        
        X = data.drop('Enterococci', axis=1)  # Adjust column name if needed
        y = data['Enterococci']  # Adjust column name if needed
        
        # Train a separate LightGBM model for each specified quantile.
        for quantile in self.quantile_levels:
            self.logger.info(f"Training LightGBM quantile model for quantile: {quantile:.2f}")
            # Create a copy of the parameters and update for quantile objective.
            params = self.params.copy()
            params['objective'] = 'quantile'
            params['alpha'] = quantile
            
            model = lgb.LGBMRegressor(**params, verbose=-1)
            model.fit(X, y,
                eval_set=[(X, y)],
                eval_metric='l1',
                eval_names=['train'],
                callbacks=[lgb.record_evaluation(eval_results)])    
            self.models[quantile] = model
        
        self.logger.info("Completed training quantile ensemble on full data.")

        # Generate out-of-fold predictions using k-fold CV.
        from sklearn.model_selection import KFold
        n_splits = 2  # Number of splits for k-fold CV
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=None)

        # Prepare an empty DataFrame to store OOF predictions.
        oof_preds = pd.DataFrame(index=data.index)

        for quantile in self.quantile_levels:
            self.logger.info(f"Generating out-of-fold predictions for quantile: {quantile:.2f}")
            params = self.params.copy()
            params['objective'] = 'quantile'
            params['alpha'] = quantile

            oof_pred = np.full(len(data), np.nan)
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train = y.iloc[train_idx]
                model_cv = lgb.LGBMRegressor(**params, verbose=-1)
                model_cv.fit(X_train, y_train)
                oof_pred[val_idx] = model_cv.predict(X_val)
            oof_preds[f"q_{quantile}"] = oof_pred

        # Combine the out-of-fold predictions with the original training data.
        # final_oof_df = data.copy()
        # final_oof_df = pd.concat([final_oof_df, oof_preds], axis=1)
        final_oof_df = oof_preds.copy()
        final_oof_df["Enterococci"] = y

        self.logger.info("Out-of-fold predictions generated successfully.")

        return final_oof_df

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions from each quantile model and return a DataFrame
        where each column corresponds to a quantile prediction.
        
        Args:
            X: Feature DataFrame for prediction.
            
        Returns:
            A DataFrame with columns named as "q_<quantile>" containing predictions.
        """
        
        if not self.models:
            self.logger.error("Attempted to predict before training the ensemble.")
            raise ValueError("No trained models available. Call train() first.")
        
        predictions = {}
        for quantile, model in self.models.items():
            col_name = f"q_{quantile}"
            predictions[col_name] = model.predict(X)
        return pd.DataFrame(predictions, index=X.index)

    def save(self, output_path: Path) -> None:
        """
        Save the entire ensemble model to a file.
        
        Args:
            output_path: Path where the model should be saved.
        """
        self.logger.info(f"Saving Probabilistic Quantile Ensemble Model to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_path)
        self.logger.info("Model saved successfully.")

    @classmethod
    def load(cls, input_path: Path) -> "ProbabilisticQuantileEnsembleModel":
        """
        Load a previously saved ensemble model.
        
        Args:
            input_path: Path to the saved model file.
        
        Returns:
            An instance of ProbabilisticQuantileEnsembleModel.
        """
        model = joblib.load(input_path)
        return model