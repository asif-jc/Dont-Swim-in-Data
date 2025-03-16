from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from termcolor import colored
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    fbeta_score
)

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

class Evaluator:
    """Base evaluator class for water quality forecasting models.
    
    This class provides the core functionality for evaluating model performance
    using various cross-validation strategies and metrics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the evaluator.
        
        Args:
            config: Evaluation configuration dictionary
        """
        self.config = config
        self.metrics = config.get("metrics", ["rmse", "sensitivity", "specificity", 
                                         "wmape_safe", "wmape_exceedance"])
        self.exceedance_threshold = config.get("exceedance_threshold", 280)
        self.precautionary_threshold = config.get("precautionary_threshold", 140)
        self.results = {}
        self.train_metrics = {}
        self.test_metrics = {}

    def evaluate_model(self, model: Any, model_name: str, X: pd.DataFrame, y: pd.Series, 
                     dataset_name: str = "test") -> Dict[str, float]:
        """Evaluate a model on a given dataset.
        
        Args:
            model: Trained model with predict method
            X: Feature DataFrame
            y: Target Series
            dataset_name: Name of the dataset for result storage
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Generate predictions
        y_pred_fold = model.predict(X)

        if (model_name == "probabilistic_framework"):
            y_pred = y_pred_fold['predictions']

        else: 
            y_pred = y_pred_fold.copy()

        # Calculate metrics
        y_pred.index = y.index
        metrics_results = self._calculate_metrics(y, y_pred)

        # Store results
        if dataset_name not in self.results:
            self.results[dataset_name] = {}
        
        self.results[dataset_name] = metrics_results
        
        # Log basic results
        logger.info(f"Evaluation on {dataset_name} dataset:")
        for metric, value in metrics_results.items():
            logger.info(f"{metric}: {value:.4f}")
            
        return metrics_results, y_pred_fold
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate all requested evaluation metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            
        Returns:
            Dictionary of metric names and values
        """
        results = {}
        
        # Create masks for safe and exceedance conditions
        true_exceedances = y_true >= self.exceedance_threshold
        true_safe = ~true_exceedances
        
        predicted_exceedances = y_pred >= self.exceedance_threshold
        predicted_safe = ~predicted_exceedances
        
        # Calculate raw counts
        TP = np.sum(np.logical_and(true_exceedances, predicted_exceedances))
        FP = np.sum(np.logical_and(true_safe, predicted_exceedances))
        TN = np.sum(np.logical_and(true_safe, predicted_safe))
        FN = np.sum(np.logical_and(true_exceedances, predicted_safe))
        
        # Store raw counts
        results["TP"] = int(TP)
        results["FP"] = int(FP)
        results["TN"] = int(TN)
        results["FN"] = int(FN)
        
        # Calculate RMSE
        if "rmse" in self.metrics:
            results["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
            
            # Calculate separate RMSE for safe and exceedance conditions
            if np.any(true_safe):
                results["rmse_safe"] = np.sqrt(mean_squared_error(
                    y_true[true_safe], y_pred[true_safe]))
            
            if np.any(true_exceedances):
                results["rmse_exceedance"] = np.sqrt(mean_squared_error(
                    y_true[true_exceedances], y_pred[true_exceedances]))
        
        # Calculate R-squared
        if "r2" in self.metrics:
            results["r2"] = r2_score(y_true, y_pred)
        
        # Calculate WMAPE (Weighted Mean Absolute Percentage Error)
        if "wmape_safe" in self.metrics and np.any(true_safe):
            results["wmape_safe"] = self._calculate_wmape(
                y_true[true_safe], y_pred[true_safe])
        
        if "wmape_exceedance" in self.metrics and np.any(true_exceedances):
            results["wmape_exceedance"] = self._calculate_wmape(
                y_true[true_exceedances], y_pred[true_exceedances])
        
        # Calculate classification metrics
        if "sensitivity" in self.metrics:
            # Use raw counts to calculate sensitivity
            if TP + FN > 0:
                results["sensitivity"] = TP / (TP + FN)
            else:
                results["sensitivity"] = 1.0
        
        if "specificity" in self.metrics:
            # Use raw counts to calculate specificity
            if TN + FP > 0:
                results["specificity"] = TN / (TN + FP)
            else:
                results["specificity"] = 1.0
            
        if "precautionary_sensitivity" in self.metrics:
            # For precautionary sensitivity, we need a different calculation
            precautionary_predictions = y_pred >= self.precautionary_threshold
            precautionary_TP = np.sum(np.logical_and(true_exceedances, precautionary_predictions))
            
            if TP + FN > 0:
                results["precautionary_sensitivity"] = precautionary_TP / (TP + FN)
            else:
                results["precautionary_sensitivity"] = 1.0
                
        return results
    
    def _calculate_wmape(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate Weighted Mean Absolute Percentage Error.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            
        Returns:
            WMAPE value as a percentage
        """
        return 100.0 * np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))
    
    def _calculate_sensitivity(self, y_true: pd.Series, y_pred: np.ndarray, 
                            threshold: float) -> float:
        """Calculate sensitivity (true positive rate).
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            threshold: Exceedance threshold
            
        Returns:
            Sensitivity value (0-1)
        """
        # True exceedances
        true_exceedances = y_true >= threshold
        
        # Predicted exceedances
        predicted_exceedances = y_pred >= threshold
        
        # True positives
        true_positives = np.logical_and(true_exceedances, predicted_exceedances)
        
        # Calculate sensitivity
        if np.sum(true_exceedances) > 0:
            return np.sum(true_positives) / np.sum(true_exceedances)
        else:
            return 1.0  # No exceedances to detect
    
    def _calculate_specificity(self, y_true: pd.Series, y_pred: np.ndarray, 
                            threshold: float) -> float:
        """Calculate specificity (true negative rate).
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            threshold: Exceedance threshold
            
        Returns:
            Specificity value (0-1)
        """
        # True safe conditions
        true_safe = y_true < threshold
        
        # Predicted safe conditions
        predicted_safe = y_pred < threshold
        
        # True negatives
        true_negatives = np.logical_and(true_safe, predicted_safe)
        
        # Calculate specificity
        if np.sum(true_safe) > 0:
            return np.sum(true_negatives) / np.sum(true_safe)
        else:
            return 1.0  # No safe conditions to identify
    
    def _calculate_precautionary_sensitivity(self, y_true: pd.Series, y_pred: np.ndarray,
                                          exceedance_threshold: float,
                                          precautionary_threshold: float) -> float:
        """Calculate precautionary sensitivity.
        
        This counts a prediction as a true positive if it exceeds the precautionary threshold
        even if the actual exceedance threshold is higher.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            exceedance_threshold: Actual exceedance threshold
            precautionary_threshold: Lower precautionary threshold
            
        Returns:
            Precautionary sensitivity value (0-1)
        """
        # True exceedances
        true_exceedances = y_true >= exceedance_threshold
        
        # Precautionary predictions
        precautionary_predictions = y_pred >= precautionary_threshold
        
        # True positives (with precautionary consideration)
        true_positives = np.logical_and(true_exceedances, precautionary_predictions)
        
        # Calculate precautionary sensitivity
        if np.sum(true_exceedances) > 0:
            return np.sum(true_positives) / np.sum(true_exceedances)
        else:
            return 1.0  # No exceedances to detect
    
    def generate_report(self, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """Generate an evaluation report with all results.
        
        Args:
            output_path: Optional path to save the report
            
        Returns:
            Dictionary containing all evaluation results
        """
        if not self.results:
            logger.warning("No evaluation results to report")
            return {}
        
        # Create JSON-serializable report
        # Convert any DataFrames to separate files or dictionaries
        serializable_results = {}
        
        # Handle nested dictionaries and DataFrames
        for key, value in self.results.items():
            if isinstance(value, pd.DataFrame):
                # Save DataFrame to CSV if output path is provided
                if output_path:
                    csv_path = Path(str(output_path).replace('.json', f'_{key}.csv'))
                    value.to_csv(csv_path, index=False)
                    serializable_results[key] = f"Saved to {csv_path.name}"
                    
                    # Add some summary statistics
                    serializable_results[f"{key}_summary"] = {
                        "shape": value.shape,
                        "columns": value.columns.tolist()
                    }
                else:
                    # If no output path, include basic statistics only
                    serializable_results[f"{key}_summary"] = {
                        "shape": value.shape,
                        "columns": value.columns.tolist()
                    }
            elif isinstance(value, dict):
                # Handle nested dictionaries
                serializable_dict = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, pd.DataFrame):
                        if output_path:
                            csv_path = Path(str(output_path).replace('.json', f'_{key}_{sub_key}.csv'))
                            sub_value.to_csv(csv_path, index=False)
                            serializable_dict[sub_key] = f"Saved to {csv_path.name}"
                        else:
                            serializable_dict[sub_key] = f"DataFrame: {sub_value.shape}"
                    else:
                        serializable_dict[sub_key] = sub_value
                serializable_results[key] = serializable_dict
            else:
                serializable_results[key] = value
        
        report = {
            "metrics": self.metrics,
            "exceedance_threshold": self.exceedance_threshold,
            "precautionary_threshold": self.precautionary_threshold,
            "results": serializable_results
        }
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=4)
            
            logger.info(f"Evaluation report saved to {output_path}")
        
        return report
    

    def performance_evaluation_old_pipeline(self, train_forecast, test_forecast, model):

        y_pred_train = train_forecast["predictions"]
        y_pred_test = test_forecast["predictions"]
        y_train = train_forecast["Enterococci"]
        y_test = test_forecast["Enterococci"]

        # print(y_train)
        # print(y_test)
        # print(y_pred_train)
        # print(y_pred_test)
        # print(model)

        # MODEL PERFORMANCE METRICS
        metrics_keys = [
            "rmse", "mae", "mape", "nrmse", "r2",
            "mae_exceedance", "mape_exceedance", "nrmse_exceedance",
            "mae_safe", "mape_safe", "nrmse_safe",
            "accuracy", "recall_safe", "recall_exceedance", "precision_safe", "precision_exceedance",
            "sensitivity", "specificity", "tn", "fp", "fn", "tp", 
            "fbeta_safe_1_5", "fbeta_exceedance_1_5", "fbeta_safe_2", "fbeta_exceedance_2", "precautionary", 
            "recall_precautionary", "weighted_mape", "weighted_mape_safe", "weighted_mape_exceedance", "r2_exceedance", "r2_safe"]
        self.train_metrics = {key: [] for key in metrics_keys}
        self.test_metrics = {key: [] for key in metrics_keys}

        self.train_metrics = self.regression_performance(y_train, y_pred_train, self.train_metrics)
        self.test_metrics = self.regression_performance(y_test, y_pred_test, self.test_metrics)

        y_train_flag, y_pred_train_flag = self.convert_target_to_flag(y_train, 280), self.convert_target_to_flag(y_pred_train, 280)
        y_test_flag, y_pred_test_flag = self.convert_target_to_flag(y_test, 280), self.convert_target_to_flag(y_pred_test, 280)

        self.train_metrics = self.classification_performance(y_train_flag, y_pred_train_flag, y_train, y_pred_train, self.train_metrics)
        self.test_metrics = self.classification_performance(y_test_flag, y_pred_test_flag, y_test, y_pred_test, self.test_metrics)

        self.display_performance(self.train_metrics, self.test_metrics, model)

    def convert_target_to_flag(self, y, threshold):
        return pd.Series(["EXCEEDANCE" if val >= threshold else "SAFE" for val in y])
    
    def regression_performance(self, y_true, y_pred, metrics):

        def log_mape(y_true, y_pred):
            y_true_log = np.log1p(y_true)
            y_pred_log = np.log1p(y_pred)

            log_APE = np.abs((y_true_log - y_pred_log) / y_true_log)
            log_MAPE = np.mean(log_APE)
            return log_MAPE
        
        def weighted_mape(y_true, y_pred):
            """Calculate Weighted Mean Absolute Percentage Error
            Weights are proportional to the true values"""
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            
            # Avoid division by zero
            mask = y_true != 0
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            
            # Calculate weights (normalized true values)
            weights = y_true / np.sum(y_true)
            
            # Calculate individual absolute percentage errors
            absolute_percentage_errors = np.abs((y_true - y_pred) / y_true)
            
            # Calculate weighted average
            wmape = np.sum(weights * absolute_percentage_errors)
            
            return wmape
        
        # Handle empty series
        if len(y_true) == 0:
            y_true = pd.Series([5])
        if len(y_pred) == 0:
            y_pred = pd.Series([5])

        y_true = np.array(y_true).flatten()  # Ensure y_true is a 1D array
        y_pred = np.array(y_pred).flatten()  # Ensure y_pred is a 1D array

        y_pred[y_pred <= 0] = 5    # Replace negative values with 0
        
        # # Replace NaN values with 10
        # y_true = np.where(np.isnan(y_true), 10, y_true)
        # y_pred = np.where(np.isnan(y_pred), 10, y_pred)

        # rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        rmse = np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))
        mae = mean_absolute_error(y_true, y_pred)
        # mape = log_mape(y_true, y_pred)
        wmape = weighted_mape(y_true, y_pred)
        mape = weighted_mape(np.log1p(y_true), np.log1p(y_pred))

        r2 = r2_score(np.log1p(y_true), np.log1p(y_pred))

        # Calculate NRMSE as a percentage
        range_y_true = np.max(np.log1p(y_true)) - np.min(np.log1p(y_true))
        if range_y_true == 0:
            nrmse = -1
        else:
            nrmse = (rmse / range_y_true) * 100

        exceedance_indices = y_true >= 280
        safe_indices = y_true < 280

        y_exceedance = y_true[exceedance_indices]
        final_predictions_exceedance = y_pred[exceedance_indices]
        y_safe = y_true[safe_indices]
        final_predictions_safe = y_pred[safe_indices]

        if len(y_exceedance) == 0 or len(final_predictions_exceedance) == 0:
            mae_exceedance = -1
            mape_exceedance = -1
            nrmse_exceedance = -1
            wmape_exceedance = -1
        else:
            mae_exceedance = mean_absolute_error(y_exceedance, final_predictions_exceedance)
            wmape_exceedance = weighted_mape(y_exceedance, final_predictions_exceedance)
            
            if (np.max(y_exceedance) - np.min(y_exceedance)) == 0:
                y_exceedance_norm = y_exceedance  # Avoid division by zero
            else:
                y_exceedance_norm = y_exceedance / (np.max(y_exceedance) - np.min(y_exceedance))

            if (np.max(final_predictions_exceedance) - np.min(final_predictions_exceedance)) == 0:
                final_predictions_exceedance_norm = final_predictions_exceedance  # Avoid division by zero
            else:
                final_predictions_exceedance_norm = final_predictions_exceedance / (np.max(final_predictions_exceedance) - np.min(final_predictions_exceedance))

            # mape_exceedance = mean_absolute_percentage_error(y_exceedance_norm, final_predictions_exceedance_norm)
            # mape_exceedance = log_mape(y_exceedance, final_predictions_exceedance)
            mape_exceedance = weighted_mape(np.log1p(y_exceedance), np.log1p(final_predictions_exceedance))

            range_y_exceedance = np.max(y_exceedance) - np.min(y_exceedance)
            if range_y_exceedance == 0:
                nrmse_exceedance = -1
            else:
                nrmse_exceedance = (np.sqrt(mean_squared_error(y_exceedance, final_predictions_exceedance)) / range_y_exceedance) * 100

        if len(y_safe) == 0 or len(final_predictions_safe) == 0:
            mae_safe = -1
            mape_safe = -1
            nrmse_safe = -1
            wmape_safe = -1
        else:
            mae_safe = mean_absolute_error(y_safe, final_predictions_safe)
            wmape_safe = weighted_mape(y_safe, final_predictions_safe)

            if (np.max(y_safe) - np.min(y_safe)) == 0:
                y_safe_norm = y_safe  # Avoid division by zero
            else:
                y_safe_norm = y_safe / (np.max(y_safe) - np.min(y_safe))

            if (np.max(final_predictions_safe) - np.min(final_predictions_safe)) == 0:
                final_predictions_safe_norm = final_predictions_safe  # Avoid division by zero
            else:
                final_predictions_safe_norm = final_predictions_safe / (np.max(final_predictions_safe) - np.min(final_predictions_safe))

            # mape_safe = mean_absolute_percentage_error(y_safe_norm, final_predictions_safe_norm)
            # mape_safe = log_mape(y_safe, final_predictions_safe)
            mape_safe = weighted_mape(np.log1p(y_safe), np.log1p(final_predictions_safe))

            range_y_safe = np.max(y_safe) - np.min(y_safe)
            if range_y_safe == 0:
                nrmse_safe = -1
            else:
                nrmse_safe = (np.sqrt(mean_squared_error(y_safe, final_predictions_safe)) / range_y_safe) * 100

        # Calculate R2 for exceedance cases
        if len(y_exceedance) > 0:
            r2_exceedance = r2_score(np.log1p(y_exceedance), np.log1p(final_predictions_exceedance))
        else:
            r2_exceedance = -1

        # Calculate R2 for safe cases
        if len(y_safe) > 0:
            r2_safe = r2_score(np.log1p(y_safe), np.log1p(final_predictions_safe))
        else:
            r2_safe = -1

        metrics["rmse"].append(rmse)
        metrics["mae"].append(mae)
        metrics["mape"].append(mape)
        metrics["nrmse"].append(nrmse)
        metrics["mae_exceedance"].append(mae_exceedance)
        metrics["mape_exceedance"].append(mape_exceedance)
        metrics["nrmse_exceedance"].append(nrmse_exceedance)
        metrics["mae_safe"].append(mae_safe)
        metrics["mape_safe"].append(mape_safe)
        metrics["nrmse_safe"].append(nrmse_safe)
        metrics["weighted_mape"].append(wmape)
        metrics["weighted_mape_safe"].append(wmape_safe)
        metrics["weighted_mape_exceedance"].append(wmape_exceedance)
        metrics["r2"].append(r2)
        metrics["r2_exceedance"].append(r2_exceedance)
        metrics["r2_safe"].append(r2_safe)

        return metrics
    

    def classification_performance(self, y_true, y_pred, y_true_regression, y_pred_regression, metrics):
        accuracy = accuracy_score(y_true, y_pred)
        recall_safe = recall_score(y_true, y_pred, pos_label="SAFE")
        recall_exceedance = recall_score(y_true, y_pred, pos_label="EXCEEDANCE")
        precision_safe = precision_score(y_true, y_pred, pos_label="SAFE")
        precision_exceedance = precision_score(y_true, y_pred, pos_label="EXCEEDANCE")
        conf_matrix = confusion_matrix(y_true, y_pred, labels=["SAFE", "EXCEEDANCE"])
        sensitivity = (recall_exceedance + recall_safe) / 2
        specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
        tn, fp, fn, tp = conf_matrix.ravel()

        fbeta_safe_1_5 = fbeta_score(y_true, y_pred, pos_label="SAFE", beta=1.5)
        fbeta_exceedance_1_5 = fbeta_score(y_true, y_pred, pos_label="EXCEEDANCE", beta=1.5)
        fbeta_safe_2 = fbeta_score(y_true, y_pred, pos_label="SAFE", beta=2)
        fbeta_exceedance_2 = fbeta_score(y_true, y_pred, pos_label="EXCEEDANCE", beta=2)

        # New metric: number of cases where true value is exceedance and predicted concentration is in warning range
        precautionary_cases = sum((y_true == "EXCEEDANCE") & (y_pred_regression >= 140) & (y_pred_regression < 280))
        # New metric: recall of precautionary predictions
        # true_positives_exceedance = sum((y_true == "EXCEEDANCE") & (y_pred_regression <= 280))
        # true_positives_exceedance2 = sum((y_true == "EXCEEDANCE") & (y_pred == "EXCEEDANCE"))
        all_actual_exceedances = sum(y_true == "EXCEEDANCE")
        recall_precautionary = (precautionary_cases + tp) / all_actual_exceedances if all_actual_exceedances > 0 else 0
        
        # TODO: PRECAUTIONARY RECALL IS NOT WORKING
        # print(precautionary_cases)
        # print(true_positives_exceedance)
        # print(all_actual_exceedances)
        # print(true_positives_exceedance2)
        # print(tp)

        metrics["accuracy"].append(accuracy)
        metrics["recall_safe"].append(recall_safe)
        metrics["recall_exceedance"].append(recall_exceedance)
        metrics["precision_safe"].append(precision_safe)
        metrics["precision_exceedance"].append(precision_exceedance)
        metrics["sensitivity"].append(sensitivity)
        metrics["specificity"].append(specificity)
        metrics["tn"].append(tn)
        metrics["fp"].append(fp)
        metrics["fn"].append(fn)
        metrics["tp"].append(tp)
        metrics["fbeta_safe_1_5"].append(fbeta_safe_1_5)
        metrics["fbeta_exceedance_1_5"].append(fbeta_exceedance_1_5)
        metrics["fbeta_safe_2"].append(fbeta_safe_2)
        metrics["fbeta_exceedance_2"].append(fbeta_exceedance_2)
        metrics["precautionary"].append(precautionary_cases + tp)
        metrics["recall_precautionary"].append(recall_precautionary)

        return metrics
    
    def display_performance(self, train_metrics, test_metrics, models):
        avg_train_rmse = round(np.mean(train_metrics["rmse"]), 4)
        avg_train_mae = round(np.mean(train_metrics["mae"]), 4)
        avg_train_mape = round(np.mean(train_metrics["mape"]), 4)
        avg_train_nrmse = round(np.mean(train_metrics["nrmse"]), 4)
        avg_train_wmape = round(np.mean(train_metrics["weighted_mape"]), 4)
        avg_train_r2 = round(np.mean(train_metrics["r2"]), 4)

        # Calculate average regression metrics for test set
        avg_test_rmse = round(np.mean(test_metrics["rmse"]), 4)
        avg_test_mae = round(np.mean(test_metrics["mae"]), 4)
        avg_test_mape = round(np.mean(test_metrics["mape"]), 4)
        avg_test_nrmse = round(np.mean(test_metrics["nrmse"]), 4)
        avg_test_wmape = round(np.mean(test_metrics["weighted_mape"]), 4)
        avg_test_r2 = round(np.mean(test_metrics["r2"]), 4)

        # Calculate average classification metrics for training set
        avg_train_accuracy = round(np.mean(train_metrics["accuracy"]), 4)
        avg_train_recall_safe = round(np.mean(train_metrics["recall_safe"]), 4)
        avg_train_recall_exceedance = round(np.mean(train_metrics["recall_exceedance"]), 4)
        avg_train_precision_safe = round(np.mean(train_metrics["precision_safe"]), 4)
        avg_train_precision_exceedance = round(np.mean(train_metrics["precision_exceedance"]), 4)
        avg_train_sensitivity = round(np.mean(train_metrics["sensitivity"]), 4)
        avg_train_specificity = round(np.mean(train_metrics["specificity"]), 4)

        avg_train_fbeta_safe_1_5 = round(np.mean(train_metrics["fbeta_safe_1_5"]), 4)
        avg_train_fbeta_exceedance_1_5 = round(np.mean(train_metrics["fbeta_exceedance_1_5"]), 4)
        avg_train_fbeta_safe_2 = round(np.mean(train_metrics["fbeta_safe_2"]), 4)
        avg_train_fbeta_exceedance_2 = round(np.mean(train_metrics["fbeta_exceedance_2"]), 4)

        # Calculate average classification metrics for test set
        avg_test_accuracy = round(np.mean(test_metrics["accuracy"]), 4)
        avg_test_recall_safe = round(np.mean(test_metrics["recall_safe"]), 4)
        avg_test_recall_exceedance = round(np.mean(test_metrics["recall_exceedance"]), 4)
        avg_test_recall_precautionary = round(np.mean(test_metrics["recall_precautionary"]), 4)
        avg_test_precision_safe = round(np.mean(test_metrics["precision_safe"]), 4)
        avg_test_precision_exceedance = round(np.mean(test_metrics["precision_exceedance"]), 4)
        avg_test_sensitivity = round(np.mean(test_metrics["sensitivity"]), 4)
        avg_test_specificity = round(np.mean(test_metrics["specificity"]), 4)

        avg_test_fbeta_safe_1_5 = round(np.mean(test_metrics["fbeta_safe_1_5"]), 4)
        avg_test_fbeta_exceedance_1_5 = round(np.mean(test_metrics["fbeta_exceedance_1_5"]), 4)
        avg_test_fbeta_safe_2 = round(np.mean(test_metrics["fbeta_safe_2"]), 4)
        avg_test_fbeta_exceedance_2 = round(np.mean(test_metrics["fbeta_exceedance_2"]), 4)

        sum_test_tp = round(np.sum(test_metrics["tp"]), 4)
        sum_test_fp = round(np.sum(test_metrics["fp"]), 4)
        sum_test_fn = round(np.sum(test_metrics["fn"]), 4)
        sum_test_tn = round(np.sum(test_metrics["tn"]), 4)
        sum_test_precautionary = round(np.sum(test_metrics["precautionary"]), 4)

        sum_train_tp = round(np.sum(train_metrics["tp"]), 4)
        sum_train_fp = round(np.sum(train_metrics["fp"]), 4)
        sum_train_fn = round(np.sum(train_metrics["fn"]), 4)
        sum_train_tn = round(np.sum(train_metrics["tn"]), 4)
        sum_train_precautionary = round(np.sum(train_metrics["precautionary"]), 4)

        # Calculate average MAE, MAPE, and NRMSE for EXCEEDANCE cases
        avg_train_mae_exceedance = round(np.mean(train_metrics["mae_exceedance"]), 4)
        avg_train_mape_exceedance = round(np.mean(train_metrics["mape_exceedance"]), 4)
        avg_train_nrmse_exceedance = round(np.mean(train_metrics["nrmse_exceedance"]), 4)
        avg_train_wmape_exceedance = round(np.mean(train_metrics["weighted_mape_exceedance"]), 4)
        avg_train_r2_exceedance = round(np.mean(train_metrics["r2_exceedance"]), 4)
        avg_train_r2_safe = round(np.mean(train_metrics["r2_safe"]), 4)

        avg_test_mae_exceedance = round(np.mean(test_metrics["mae_exceedance"]), 4)
        avg_test_mape_exceedance = round(np.mean(test_metrics["mape_exceedance"]), 4)
        avg_test_nrmse_exceedance = round(np.mean(test_metrics["nrmse_exceedance"]), 4)
        avg_test_wmape_exceedance = round(np.mean(test_metrics["weighted_mape_exceedance"]), 4)
        avg_test_r2_exceedance = round(np.mean(test_metrics["r2_exceedance"]), 4)
        avg_test_r2_safe = round(np.mean(test_metrics["r2_safe"]), 4)

        # Calculate average MAE, MAPE, and NRMSE for SAFE cases
        avg_train_mae_safe = round(np.mean(train_metrics["mae_safe"]), 4)
        avg_train_mape_safe = round(np.mean(train_metrics["mape_safe"]), 4)
        avg_train_nrmse_safe = round(np.mean(train_metrics["nrmse_safe"]), 4)
        avg_train_wmape_safe = round(np.mean(train_metrics["weighted_mape_safe"]), 4)

        avg_test_mae_safe = round(np.mean(test_metrics["mae_safe"]), 4)
        avg_test_mape_safe = round(np.mean(test_metrics["mape_safe"]), 4)
        avg_test_nrmse_safe = round(np.mean(test_metrics["nrmse_safe"]), 4)
        avg_test_wmape_safe = round(np.mean(test_metrics["weighted_mape_safe"]), 4)


        print('\n')
        print(colored("---------------------------------", 'cyan'))
        print(colored(f"FORECAST PERFORMANCE (MODEL: {models})", 'red'))
        print(colored("---------------------------------", 'cyan'))
        # print(f"{colored('PERFORMANCE EVALUATED ON HOLDOUT PERIOD:', 'light_cyan')} {colored(PIPELINE_CONFIG.EVALUATE_TIMESERIES_HOLDOUT_START, 'light_red')}  {colored('-->', 'light_red')}  {colored(PIPELINE_CONFIG.EVALUATE_TIMESERIES_HOLDOUT_END, 'light_red')}")

        print(colored("-----", 'cyan'))
        print(colored(f"TRAINING PERFORMANCE (2013-10-01 00:00:00 - 2023-10)", 'light_cyan'))
        print('\n')
        print(colored("Accuracy:", 'green'), colored(avg_train_accuracy, 'green'))
        print(colored("Sensitivity/Recall (EXCEEDANCE):", 'green'), colored(avg_train_recall_exceedance, 'green'))
        print(colored("Specificity/Recall (SAFE):", 'green'), colored(avg_train_specificity, 'green'))
        print(colored("Precision (SAFE):", 'green'), colored(avg_train_precision_safe, 'green'))
        print(colored("Precision (EXCEEDANCE):", 'green'), colored(avg_train_precision_exceedance, 'green'))
        print(colored("True Positives (TRUE: EXCEEDANCE, MODEL: EXCEEDANCE):", 'blue'), colored(sum_train_tp, 'blue'))
        print(colored("False Negatives (TRUE: EXCEEDANCE, MODEL: SAFE):", 'blue'), colored(sum_train_fn, 'blue'))
        print(colored("True Negatives (TRUE: SAFE, MODEL: SAFE):", 'blue'), colored(sum_train_tn, 'blue'))
        print(colored("False Positives (TRUE: SAFE, MODEL: EXCEEDANCE):", 'blue'), colored(sum_train_fp, 'blue'))
        print(colored("Precautionary Cases (TRUE: EXCEEDANCE, MODEL: EXCEEDANCE/WARNING):", 'blue'), colored(sum_train_precautionary, 'blue'))
        print('\n')
        print(colored("Weighted MAPE:", 'blue'), colored(f"{avg_train_wmape}", 'blue'))
        print(colored("Log-Weighted MAPE:", 'blue'), colored(avg_train_mape, 'blue'))
        print(colored("MAE:", 'blue'), colored(avg_train_mae, 'blue'))
        print(colored("Weighted MAPE (EXCEEDANCE):", 'blue'), colored(f"{avg_train_wmape_exceedance}", 'blue'))
        print(colored("Log-Weighted MAPE (EXCEEDANCE):", 'blue'), colored(avg_train_mape_exceedance, 'blue'))
        print(colored("MAE (EXCEEDANCE):", 'blue'), colored(avg_train_mae_exceedance, 'blue'))
        print(colored("Weighted MAPE (SAFE):", 'blue'), colored(f"{avg_train_wmape_safe}", 'blue'))
        print(colored("Log-Weighted MAPE (SAFE):", 'blue'), colored(avg_train_mape_safe, 'blue'))
        print(colored("MAE (SAFE):", 'blue'), colored(avg_train_mae_safe, 'blue'))
        print(colored("-----", 'cyan'))
        print('\n')

        print(colored("-----", 'cyan'))
        print(colored(f"TEST PERFORMANCE (2021-10 - 2024-10))", 'light_cyan'))
        print('\n')
        print(colored("Accuracy:", 'green'), colored(avg_test_accuracy, 'green'))
        print(colored("Sensitivity/Recall (EXCEEDANCE):", 'green'), colored(avg_test_recall_exceedance, 'green'))
        print(colored("Specificity/Recall (SAFE):", 'green'), colored(avg_test_specificity, 'green'))
        print(colored("Specificity/Recall (PRECAUTIONARY):", 'green'), colored(avg_test_recall_precautionary, 'green'))
        print(colored("Precision (SAFE):", 'green'), colored(avg_test_precision_safe, 'green'))
        print(colored("Precision (EXCEEDANCE):", 'green'), colored(avg_test_precision_exceedance, 'green'))
        print(colored("F-2 Score:", 'green'), colored(avg_test_fbeta_exceedance_2, 'green'))
        print(colored("True Positives (TRUE: EXCEEDANCE, MODEL: EXCEEDANCE):", 'blue'), colored(sum_test_tp, 'blue'))
        print(colored("False Negatives (TRUE: EXCEEDANCE, MODEL: SAFE):", 'blue'), colored(sum_test_fn, 'blue'))
        print(colored("True Negatives (TRUE: SAFE, MODEL: SAFE):", 'blue'), colored(sum_test_tn, 'blue'))
        print(colored("False Positives (TRUE: SAFE, MODEL: EXCEEDANCE):", 'blue'), colored(sum_test_fp, 'blue'))
        print(colored("Precautionary Cases (TRUE: EXCEEDANCE, MODEL: EXCEEDANCE/WARNING):", 'blue'), colored(sum_test_precautionary, 'blue'))
        print('\n')
        
        print(colored("RMSE:", 'blue'), colored(f"{avg_test_rmse}", 'blue'))
        print(colored("NRMSE:", 'blue'), colored(f"{avg_test_nrmse}", 'blue'))
        print(colored("Weighted MAPE:", 'blue'), colored(f"{avg_test_wmape}", 'blue'))
        print(colored("Log-Weighted MAPE:", 'blue'), colored(avg_test_mape, 'blue'))
        print(colored("MAE:", 'blue'), colored(avg_test_mae, 'blue'))
        print(colored("log-R2:", 'blue'), colored(f"{avg_test_r2}", 'blue'))
        print(colored("Weighted MAPE (EXCEEDANCE):", 'blue'), colored(f"{avg_test_wmape_exceedance}", 'blue'))
        print(colored("Log-Weighted MAPE (EXCEEDANCE):", 'blue'), colored(avg_test_mape_exceedance, 'blue'))
        print(colored("MAE (EXCEEDANCE):", 'blue'), colored(avg_test_mae_exceedance, 'blue'))
        print(colored("Weighted MAPE (SAFE):", 'blue'), colored(f"{avg_test_wmape_safe}", 'blue'))
        print(colored("Log-Weighted MAPE (SAFE):", 'blue'), colored(avg_test_mape_safe, 'blue'))
        print(colored("MAE (SAFE):", 'blue'), colored(avg_test_mae_safe, 'blue'))
        print(colored("NRMSE (EXCEEDANCE):", 'blue'), colored(avg_test_nrmse_exceedance, 'blue'))
        print(colored("NRMSE (SAFE):", 'blue'), colored(avg_test_nrmse_safe, 'blue'))
        print(colored("R2 (EXCEEDANCE):", 'blue'), colored(avg_test_r2_exceedance, 'blue'))
        print(colored("R2 (SAFE):", 'blue'), colored(avg_test_r2_safe, 'blue'))
        print(colored("log-R2:", 'blue'), colored(f"{avg_test_r2}", 'blue'))
        
        print(colored("-----", 'cyan'))

        print(colored("------------------------------------------------------------------", 'cyan'))
        print('\n')
