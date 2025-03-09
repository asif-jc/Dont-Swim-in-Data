# src/pipeline/main_pipeline.py
import argparse
import logging
from pathlib import Path
import yaml
import sys
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent  
sys.path.append(str(project_root))

from src.data.data_loader import DataLoader
from src.data.preprocessing import Preprocessor
from src.data.feature_engineering import FeatureEngineer
from src.models.model_factory import ModelFactory
from src.evaluation.time_series_evaluator import TimeSeriesEvaluator
from src.evaluation.evaluator import Evaluator
from src.utils.logging import setup_logger
# from src.visualization.forecast_plots import create_forecast_visualizations
from src.visualization.interactive_plots import create_interactive_forecast_plot
from src.visualization.dashboard import generate_dashboard

from src.config.paths import (
    TRAINING_DATA_PATH,
    MAIN_CONFIG_PATH
)

logger = setup_logger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Water Quality Forecasting Pipeline")
    parser.add_argument("--config", type=str, default="config/main_config.yaml", 
                        help="Path to configuration file")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "predict", "all"], 
                        default="all", help="Pipeline execution mode")
    return parser.parse_args()


def run_pipeline(config_path: str, mode: str):
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Starting pipeline in {mode} mode with config from {config_path}")

    # Initialize components
    data_loader = DataLoader(config_path)
    preprocessor = Preprocessor(config_path)
    feature_engineer = FeatureEngineer(config_path)
    model_factory = ModelFactory(config)
    
    # Load and process data
    data = data_loader.load_processed_data(TRAINING_DATA_PATH)
    data = preprocessor.clean_data(data)
    data["Enterococci"] = preprocessor.set_max_target_value(data["Enterococci"])
    data = feature_engineer.engineer_features(data)
    data = preprocessor.transform_catergorical_variable_type(data)

    logger.info(f"Processed data: {data.shape} rows, {data.columns.size} columns")
    
    # Get models to process based on config
    model_names = config["pipeline"].get("models_to_process")
    logger.info(f"Processing models: {', '.join(model_names)}")
    
    # Dictionary to store trained models
    trained_models = {}
    train_forecast = {}
    test_forecast = {}

    if mode in ["train", "all"]:
        # Train all specified models
        logger.info("Training models...")
        
        for model_name in model_names:
            logger.info(f"Training {model_name}...")
            model = model_factory.get_model(model_name)
            
            # Train the model
            model.train(data)
            trained_models[model_name] = model
            
            # Save the model
            output_path = Path("models") / f"{model_name}.joblib"
            model.save(output_path)
            
            logger.info(f"Model {model_name} trained and saved to {output_path}")
    
    if mode in ["evaluate", "all"]:
        # Create time-series evaluator
        ts_evaluator = TimeSeriesEvaluator(config["evaluation"])
        
        # Evaluate all specified models
        logger.info(f"Evaluating models: {', '.join(model_names)}")
        evaluation_results = {}
        
        for model_name in model_names:
            logger.info(f"Evaluating {model_name} with time-series cross-validation")

            results, test_forecast_tscv, train_forecast_tscv = ts_evaluator.evaluate_with_tscv(
                model_factory, 
                data,
                model_name,
                date_column=config["data"].get("date_column", "DateTime"),
                target_column=config["data"].get("target_column", "Enterococci")
            )
            
            evaluation_results[model_name] = results
            test_forecast[model_name], train_forecast[model_name]  = test_forecast_tscv, train_forecast_tscv

        # Generate and save the report
        report_path = Path(config["evaluation"].get("report_path", "reports/evaluation_results.json"))
        ts_evaluator.generate_report(report_path)

        logger.info(f"Evaluation complete. Report saved to {report_path}")

    

    if mode in ["predict", "all"]:
        # Get prediction configuration
        pred_config = config.get("prediction", {})
        
        # Load prediction data and process it
        pred_data_path = Path(pred_config.get("data_path", config["data"]["prediction_data_path"]))
        logger.info(f"Loading prediction data from {pred_data_path}")
        
        pred_data = data_loader.load_processed_data(pred_data_path)
        pred_data = preprocessor.clean_data(pred_data)
        pred_data = feature_engineer.engineer_features(pred_data)
        pred_data["Enterococci"] = preprocessor.set_max_target_value(pred_data["Enterococci"])
        pred_data = preprocessor.transform_catergorical_variable_type(pred_data)
        
        # Separate features and target (if present)
        target_column = config["data"].get("target_column", "Enterococci")
        date_column = config["data"].get("date_column", "DateTime")
        site_column = config["data"].get("site_column", "SITE_NAME")
        
        # Create a copy of the prediction data to store all model predictions
        results_df = pred_data.copy()
        
        if target_column in results_df.columns:
            has_target = True
            X_pred = pred_data.drop(columns=[target_column])
            y_true = pred_data[target_column]
            logger.info(f"Prediction data contains target column, will evaluate predictions")
        else:
            has_target = False
            X_pred = pred_data
            logger.info(f"Prediction data does not contain target column, only generating predictions")
        
        # Generate predictions for each model
        for model_name in model_names:
            logger.info(f"Generating predictions for {model_name}")
            
            # Load or use already trained model
            if model_name in trained_models:
                model = trained_models[model_name]
                logger.info(f"Using already trained {model_name} model")
            else:
                # Load saved model
                model_path = Path(pred_config.get(f"{model_name}_model_path", 
                                               f"models/{model_name}.joblib"))
                
                logger.info(f"Loading {model_name} model from {model_path}")
                
                try:
                    # Add loading logic for different model types
                    if model_name == "lightgbm":
                        from src.models.benchmarks.lightgbm_models import LightGBMModel
                        model = LightGBMModel.load(model_path)
                    if model_name == "probabilistic_framework":
                        from src.models.probabilistic.main_probabilistic import ProbabilisticForecastingModel
                        model = ProbabilisticForecastingModel.load(model_path)
                    if model_name == "matrix_decomposition_framework":
                        from src.models.matrix_decomp.matrix_decomposition_model import MatrixDecompositionModel
                        model = MatrixDecompositionModel.load(model_path)
                    # Add other model loaders as needed
                    else:
                        model = model_factory.get_model(model_name)
                        # Could add more specific loading logic here
                except Exception as e:
                    logger.error(f"Failed to load {model_name} model: {e}")
                    continue
            
            # Generate predictions
            try:
                predictions = model.predict(X_pred)
                
                # Add predictions to results dataframe with model-specific column name
                results_df[f"{model_name}_predictions"] = predictions
                
                logger.info(f"Generated predictions for {model_name}")
                
                # Calculate metrics if target is available
                if has_target:
                    from src.evaluation.evaluator import Evaluator
                    evaluator = Evaluator(config["evaluation"])
                    metrics = evaluator._calculate_metrics(y_true, predictions)
                    
                    logger.info(f"{model_name} prediction performance metrics:")
                    for metric, value in metrics.items():
                        logger.info(f"{metric}: {value:.4f}")
                
            except Exception as e:
                logger.error(f"Error generating predictions for {model_name}: {e}")
        
        # Save combined predictions
        output_path = Path(pred_config.get("output_path", "results/predictions.csv"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_df.to_csv(output_path, index=False)
        logger.info(f"All predictions saved to {output_path}")
        
        # Generate standard visualizations
        if pred_config.get("generate_visualizations", True):
            visualization_path = Path(pred_config.get("visualization_path", "results/visualizations"))
            visualization_path.mkdir(parents=True, exist_ok=True)
            
            # logger.info(f"Generating standard forecast visualizations in {visualization_path}")
            # vis_files = create_forecast_visualizations(results_df, visualization_path)
            # logger.info(f"Created {len(vis_files)} visualization files")
        
        # Generate interactive visualizations
        if pred_config.get("generate_interactive_visualizations", True):
            interactive_path = Path(pred_config.get("interactive_visualization_path", "results/interactive"))
            interactive_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Generating interactive forecast visualizations")
            
            # Split data into training and test sets if needed
            test_data = results_df.copy()
            train_data = None

            test_forecast_point = pd.DataFrame()
            for model_name in model_names:
                test_forecast_point[model_name] = test_forecast[f"{model_name}"]["predictions"]

            test_forecast_point["DateTime"] = results_df["DateTime"]
            test_forecast_point["SITE_NAME"] = results_df["SITE_NAME"]
            test_forecast_point["Enterococci"] = results_df["Enterococci"]            
            
            # If there's training data specified, load it
            if "training_prediction_data_path" in pred_config:
                train_path = Path(pred_config.get("training_prediction_data_path"))
                if train_path.exists():
                    train_data = pd.read_csv(train_path)
                    # Process training data if needed
                    train_data = preprocessor.clean_data(train_data)
                    train_data = feature_engineer.engineer_features(train_data)
                    train_data = preprocessor.transform_catergorical_variable_type(train_data)
                    logger.info(f"Loaded training predictions from {train_path}")
            
            # Create the visualization
            html_path = interactive_path / "forecast_comparison.html"
            create_interactive_forecast_plot(
                test_data=test_forecast_point,
                train_data=train_data,
                output_path=html_path,
                date_column=date_column,
                site_column=site_column,
                target_column=target_column
            )

            app = generate_dashboard(test_forecast)

            logger.info(f"Interactive visualization saved to {html_path}")

    return app


if __name__ == "__main__":
    args = parse_args()
    # run_pipeline(args.config, args.mode)
    app = run_pipeline(MAIN_CONFIG_PATH, "all")
    
    logger.info("Dashboard generated, starting server...")
    app.run_server(debug=True, port=8050)