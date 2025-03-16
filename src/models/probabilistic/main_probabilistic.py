import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
import joblib
import sys
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
import os
from mapie.quantile_regression import MapieQuantileRegressor

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

# Import the quantile ensemble model from our quantile modeling module
from src.models.probabilistic.quantile_modeling import ProbabilisticQuantileEnsembleModel
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

class ProbabilisticForecastingModel:
    """
    Main model class for the Probabilistic Forecasting Framework.
    
    This model first trains an ensemble of LightGBM quantile regression models (stage 1).
    It then (optionally) uses a meta-learner to produce a point forecast and calibrates the prediction 
    intervals (stages 2 and 3). Currently, the meta-learner and calibration are placeholders; for 
    now, the point forecast is derived from the median quantile.
    
    This class implements the standard interface (train, predict, save, load) so that it integrates seamlessly
    with the main pipeline.
    """
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        
        # Data settings from configuration
        # self.target_column = config["data"].get("target_column", "Enterococci")
        # self.feature_columns = config["data"].get("feature_columns", None)

        self.target_column = None
        self.feature_columns = None
    
        # Initialize the quantile ensemble component using configuration settings
        self.quantile_ensemble = ProbabilisticQuantileEnsembleModel(config)
        
        # Placeholders for meta-learner and calibration components
        self.meta_learner = self.config["models"]["probabilistic_framework"].get("meta_learner")
        self.calibration_params = None

    def train(self, data: pd.DataFrame) -> None:
        """
        Train the probabilistic forecasting model.
        
        This includes:
          - Extracting features and target in the same manner as the LightGBM baseline.
          - Adding a dummy "DateTime" column as required.
          - Training the quantile ensemble.
        
        Args:
            data: Training data as a DataFrame.
        """
        self.logger.info("Starting training of Probabilistic Forecasting Model.")
        data.to_csv("data/processed/training_data.csv", index=False)

        # Extract features and target following the LightGBM benchmark pattern.
        X = data.drop('Enterococci', axis=1)
        y = data['Enterococci']
        X["DateTime"] = 0
        
        # Recombine features and target to form a processed DataFrame.
        self.training_data = pd.concat([X, y], axis=1)
        
        # Train the quantile ensemble on the processed data.
        self.training_oof_quantile_forecast = self.quantile_ensemble.train(self.training_data)

        self.logger.info("Probabilistic Forecasting Model training complete.")

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions using the probabilistic forecasting model.
        
        This method:
          - Adds a dummy "DateTime" column as in the baseline.
          - Obtains quantile predictions from the ensemble.
          - Uses the median quantile as the point forecast.
        
        Args:
            X: Feature DataFrame for prediction.
        
        Returns:
            A DataFrame containing quantile predictions and a column 'point_forecast'.
        """
        self.logger.info("Generating predictions using the Probabilistic Forecasting Model.")
        
        # Mimic the baseline by adding a "DateTime" column.
        X = X.copy()  # Avoid modifying the original DataFrame.
        X["DateTime"] = 0
        
        # Obtain quantile predictions from the ensemble
        quantile_preds = self.quantile_ensemble.predict(X)
        
        # Apply enterococci constraints (e.g., non-negativity, upper bounds)
        quantile_preds = self.apply_enterococci_constraints(quantile_preds)
        self.training_oof_quantile_forecast = self.apply_enterococci_constraints(self.training_oof_quantile_forecast)
        
        # Calibrate prediction intervals
        lower_prediction_interval, upper_prediction_interval  = self.calibrate_intervals(quantile_preds, X)
        # print(lower_prediction_interval)
        # print(upper_prediction_interval)
        
        # Generate point forecast using the meta-learner
        if self.meta_learner:
            point_forecast = self.apply_meta_learner(self.training_oof_quantile_forecast, quantile_preds)
        else:
            point_forecast = self.apply_average_quantile(quantile_preds)

        quantile_preds = self.monotonic_sort_quantiles(quantile_preds, quantile_preds.columns)
        
        # Append point forecast to quantile predictions
        results = quantile_preds.copy()
        results["predictions"] = point_forecast
        
        # Ensure final results also satisfy enterococci constraints
        results = self.apply_enterococci_constraints(results)
        
        return results

    
    def apply_meta_learner(self, train_quantile_preds: pd.DataFrame, test_quantile_preds: pd.DataFrame) -> pd.Series:
        """
        Combine the quantile predictions using a meta-learner to produce a point forecast.
        Future implementation: Use a gradient boosting model or another ensemble method to combine
        quantile predictions.
        
        Currently, as a placeholder, we use the median of the quantile predictions.
        """
        self.logger.info("Training meta-learner for point forecast.")

        # Prepare meta-learning data.
        X_meta = train_quantile_preds.drop('Enterococci', axis=1)
        y_meta = train_quantile_preds['Enterococci']

        # Check if tuning is enabled via config.
        if self.config.get("meta_learner_tune", False):
            self.logger.info("Performing grid search tuning for meta-learner.")
            param_grid = {
                "n_estimators": [100, 150, 180, 200, 250],
                "learning_rate": [0.01, 0.05, 0.087, 0.1, 0.15],
                "num_leaves": [20, 30, 40, 50, 60],
                "min_data_in_leaf": [3, 5, 10, 20],
                "colsample_bytree": [0.7, 0.8, 0.9, 0.958, 1.0],
                "reg_lambda": [0.1, 0.5, 0.654, 1.0, 2.0]
            }
            grid_search = GridSearchCV(
                estimator=lgb.LGBMRegressor(verbose=-1),
                param_grid=param_grid,
                cv=5,
                scoring="neg_mean_absolute_error",
                n_jobs=-1
            )
            grid_search.fit(X_meta, y_meta)
            best_params = grid_search.best_params_
            self.logger.info(f"Best meta-learner parameters: {best_params}")
            self.meta_learner = grid_search.best_estimator_
        else:
            # Use parameters from config or default values.
            meta_params = self.config.get("meta_learner_params", {
                "n_estimators": 180,
                "learning_rate": 0.08721057029770096,
                "num_leaves": 40,
                "min_data_in_leaf": 5,
                "colsample_bytree": 0.9587602766121369,
                "reg_lambda": 0.6540961177848345,
                "boosting_type": "gbdt"
            })
            self.logger.info(f"Using meta-learner parameters: {meta_params}")
            self.meta_learner = lgb.LGBMRegressor(**meta_params, verbose=-1)
            self.meta_learner.fit(X_meta, y_meta)
        
        # Predict using the meta-learner.
        meta_learn_preds = self.meta_learner.predict(test_quantile_preds)
        meta_learn_preds = pd.Series(meta_learn_preds, index=test_quantile_preds.index, name="predictions")
        
        # Apply enterococci constraints (ensuring domain-specific output bounds).
        meta_learn_preds = self.apply_enterococci_constraints(meta_learn_preds)
        
        return meta_learn_preds
    
    def apply_average_quantile(self, quantile_preds: pd.DataFrame) -> pd.Series:
        """
        Combine the quantile predictions using a meta-learner to produce a point forecast.
        Future implementation: Use a gradient boosting model or another ensemble method to combine
        quantile predictions.
        
        Currently, as a placeholder, we use the median of the quantile predictions.
        """
        self.logger.info("Applying meta-learner (placeholder): using median of quantile predictions.")
        return quantile_preds.median(axis=1)

    def calibrate_intervals(self, quantile_preds: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Calibrate the prediction intervals to ensure proper coverage (e.g., 90% coverage).
        Future implementation: Apply Conformalized Quantile Regression (CQR) or similar calibration.
        
        Currently, as a placeholder, we return the quantile predictions unchanged.
        """
        self.logger.info("Calibrating prediction intervals")

        base_model = self.quantile_ensemble.models[0.8]

        X_train = self.training_data.drop('Enterococci', axis=1)
        y_train = self.training_data['Enterococci']
        
        # Create prediction intervals with symmetric alpha
        mapie = MapieQuantileRegressor(
            estimator=base_model,
            method="quantile",
            cv='split'
        )
        
        # Fit once
        mapie.fit(X_train, y_train)

        # Get predictions with proper confidence level
        # alpha=0.1 for 90% prediction interval (0.05 on each side)
        point_pred_test, intervals_test = mapie.predict(X_test, alpha=0.10)
        point_pred_train, intervals_train = mapie.predict(X_train, alpha=0.10)
        
        # Extract bounds
        lower_pred_test = intervals_test[:, 0, 0]  # Lower bound
        upper_pred_test = intervals_test[:, 1, 0]  # Upper bound
        
        lower_pred_train = intervals_train[:, 0, 0]  # Lower bound
        upper_pred_train = intervals_train[:, 1, 0]  # Upper bound


        return lower_pred_test, upper_pred_test
        
    def apply_enterococci_constraints(self, predictions):
        # Apply minimum value of 5 MPN/100mL
        constrained = np.maximum(predictions, 5)
        # Round to nearest integer
        constrained = np.round(constrained)
        return constrained
    
    def monotonic_sort_quantiles(self, df, quantile_columns):
        for index, row in df.iterrows():
            sorted_values = sorted([row[quantile] for quantile in quantile_columns])
            for i, quantile in enumerate(quantile_columns):
                df.at[index, quantile] = sorted_values[i]
        return df

    def save(self, output_path: Path) -> None:
        """
        Save the probabilistic forecasting model to disk.
        
        Args:
            output_path: Path where the model should be saved.
        """
        self.logger.info(f"Saving Probabilistic Forecasting Model to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_path)
        self.logger.info("Model saved successfully.")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ProbabilisticForecastingModel":
        """
        Lower-level training method that accepts features and target separately.
        
        Mimics the LightGBMModel.fit() behavior by adding a dummy "DateTime" column,
        then recombining features and target before training the quantile ensemble.
        
        Args:
            X: Features DataFrame.
            y: Target variable Series.
            
        Returns:
            Self reference for method chaining.
        """
        self.logger.info("Fitting Probabilistic Forecasting Model using provided X and y.")
        
        # Save the feature names
        self.feature_columns = X.columns.tolist()
        
        # Mimic the baseline by adding a "DateTime" column.
        X = X.copy()
        X["DateTime"] = 0
        
        # Recombine features and target to form a processed DataFrame.
        processed_data = pd.concat([X, y], axis=1)
        
        # Train the quantile ensemble on the processed data.
        self.quantile_ensemble.train(processed_data)
        
        self.logger.info("Probabilistic Forecasting Model fit complete.")
        return self

    @classmethod
    def load(cls, input_path: Path) -> "ProbabilisticForecastingModel":
        """
        Load a previously saved probabilistic forecasting model.
        
        Args:
            input_path: Path to the saved model file.
            
        Returns:
            An instance of ProbabilisticForecastingModel.
        """
        model = joblib.load(input_path)
        return model