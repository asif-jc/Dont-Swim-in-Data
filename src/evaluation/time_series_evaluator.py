from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd

from src.evaluation.evaluator import Evaluator
from src.evaluation.cross_validation import TimeSeriesCV
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

class TimeSeriesEvaluator(Evaluator):
    """Evaluator for time-series cross-validation.
    
    This class extends the base Evaluator to implement time-series specific
    evaluation strategies.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the time-series evaluator.
        
        Args:
            config: Evaluation configuration dictionary
        """
        super().__init__(config)
        self.tscv = TimeSeriesCV(config.get("cross_validation", {}))

    def evaluate_with_tscv(self, model_factory, data: pd.DataFrame, 
                         model_name: str, date_column: str = "sampling_date",
                         target_column: str = "enterococci") -> Dict[str, Any]:
        """Evaluate a model using time-series cross-validation.
        
        Args:
            model_factory: Factory to create model instances
            data: DataFrame containing features and target
            model_name: Name of the model to evaluate
            date_column: Name of the column containing dates
            target_column: Name of the column containing the target variable
            
        Returns:
            Dictionary containing evaluation results for each fold
        """
        # Get train-test splits
        splits = self.tscv.split(data, date_column)
        
        test_fold_results = {}
        train_fold_results = {}
        test_predictions = pd.DataFrame()
        train_predictions = pd.DataFrame()
        
        for i, (train_data, test_data) in enumerate(splits):
            fold_name = f"fold_{i+1}"
            logger.info(f"Evaluating {model_name} on {fold_name}")
            
            # Split features and target
            X_train = train_data.drop(columns=[target_column]).reset_index(drop=True)
            y_train = train_data[target_column].reset_index(drop=True)
            
            X_test = test_data.drop(columns=[target_column]).reset_index(drop=True)
            y_test = test_data[target_column].reset_index(drop=True)
            datetime_test, datetime_train = X_test["DateTime"], X_train["DateTime"]
            
            # Create and train the model
            model = model_factory.get_model(model_name)

            model.train(pd.concat([X_train, y_train], axis=1))
            
            # Evaluate the model

            test_fold_results[fold_name], y_pred_test = self.evaluate_model(
                model, model_name, X_test, y_test, dataset_name=fold_name)
            
            train_fold_results[fold_name], y_pred_train = self.evaluate_model(
                model, model_name, X_train, y_train, dataset_name=fold_name)
            
            y_pred_test = pd.DataFrame(y_pred_test)
            y_pred_test["DateTime"], y_pred_test["SITE_NAME"], y_pred_test["Enterococci"] = datetime_test, X_test["SITE_NAME"], y_test

            y_pred_train = pd.DataFrame(y_pred_train)
            y_pred_train["DateTime"], y_pred_train["SITE_NAME"], y_pred_train["Enterococci"] = datetime_train, X_train["SITE_NAME"], y_train

            test_predictions = pd.concat([test_predictions, y_pred_test], axis=0)
            train_predictions = pd.concat([train_predictions, y_pred_train], axis=0)

        train_predictions.reset_index(drop=True, inplace=True)
        test_predictions.reset_index(drop=True, inplace=True)

        # Calculate aggregated metrics across all folds
        y_true = test_predictions[target_column]
        y_pred = test_predictions['predictions']
        
        overall_results = self._calculate_metrics(y_true, y_pred)
        test_fold_results['overall'] = overall_results
        
        # Update results
        self.results = {
            'fold_results': test_fold_results,
            'fold_predictions': test_predictions
        }
        
        # Log overall results
        logger.info(f"Overall evaluation results for {model_name}:")
        for metric, value in overall_results.items():
            logger.info(f"{metric}: {value:.4f}")

        self.test_predictions = test_predictions
        self.train_predictions = train_predictions

        # model performance
        self.performance_evaluation_old_pipeline(train_predictions, test_predictions, model)

        return self.results, self.test_predictions, self.train_predictions
        

    
    def evaluate_with_nested_tscv(self, model_factory, data: pd.DataFrame,
                                model_name: str, date_column: str = "sampling_date",
                                target_column: str = "enterococci") -> Dict[str, Any]:
        """Evaluate a model using nested time-series cross-validation.
        
        Args:
            model_factory: Factory to create model instances
            data: DataFrame containing features and target
            model_name: Name of the model to evaluate
            date_column: Name of the column containing dates
            target_column: Name of the column containing the target variable
            
        Returns:
            Dictionary containing evaluation results for each fold
        """
        # Get nested train-validation-test splits
        nested_splits = self.tscv.nested_split(data, date_column)
        
        fold_results = {}
        all_predictions = []
        
        for i, (train_data, val_data, test_data) in enumerate(nested_splits):
            fold_name = f"fold_{i+1}"
            logger.info(f"Evaluating {model_name} on {fold_name} with nested validation")
            
            # Split features and target for all sets
            X_train = train_data.drop(columns=[target_column])
            y_train = train_data[target_column]
            
            X_val = val_data.drop(columns=[target_column])
            y_val = val_data[target_column]
            
            X_test = test_data.drop(columns=[target_column])
            y_test = test_data[target_column]
            
            # Create the model
            model = model_factory.get_model(model_name)
            
            # Here you would typically perform hyperparameter optimization using validation data
            combined_train_data = pd.concat([train_data, val_data], ignore_index=True)
            X_combined = combined_train_data.drop(columns=[target_column])
            y_combined = combined_train_data[target_column]
            
            # Train the model on combined data
            model.fit(X_combined, y_combined)

            training_data = pd.concat([X_combined, y_combined], axis=1)
            training_data.to_csv("data/processed/training_data.csv", index=False)
            
            # Evaluate on test data
            fold_results[fold_name] = self.evaluate_model(
                model, X_test, y_test, dataset_name=fold_name)
            
            # Store predictions for later analysis
            test_data['predictions'] = model.predict(X_test)
            all_predictions.append(test_data)
            
        # Combine all predictions
        all_predictions_df = pd.concat(all_predictions, ignore_index=True)
        
        # Calculate aggregated metrics across all folds
        y_true = all_predictions_df[target_column]
        y_pred = all_predictions_df['predictions']
        
        overall_results = self._calculate_metrics(y_true, y_pred)
        fold_results['overall'] = overall_results
        
        # Update results
        self.results = {
            'fold_results': fold_results,
            'fold_predictions': all_predictions_df
        }
        
        # Log overall results
        logger.info(f"Overall evaluation results for {model_name} with nested CV:")
        for metric, value in overall_results.items():
            logger.info(f"{metric}: {value:.4f}")
            
        return self.results