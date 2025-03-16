import argparse
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
import joblib
import sys
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

# Set project root and add it to sys.path if needed.
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

from src.models.matrix_decomp.matrix_decomposition_model import MatrixDecompositionModel
from src.utils.logging import setup_logger
from src.data.preprocessing import Preprocessor

logger = setup_logger(__name__)

class MatrixDecompositionFramework:
    """
    Main model class for the Matrix Decomposition Framework.
    
    This framework decomposes the input water quality data into latent temporal (W) and spatial (H) factors using NMF,
    trains temporal and spatial regression models to predict these latent factors from auxiliary features,
    and reconstructs the forecasted water quality matrix.
    
    Standard methods (train, predict, save, load) allow seamless integration with the pipeline.
    """
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        
        # Data configuration
        self.target_column = config["data"].get("target_column", "Enterococci")
        # Features to pivot: assume data contains 'DateTime' and 'SITE_NAME'
        self.date_column = config["data"].get("date_column", "DateTime")
        self.site_column = config["data"].get("site_column", "SITE_NAME")
        # Features for regression models:
        self.time_features = []
        self.site_features = ["SITE_NAME", "Harbour", "Latitude", "Longitude", "Shallowness", "Soil_type", "Catchment_slope", "Landcover_catchment", "watercraft_use", "sewage_discharge_beach", 
                              "high_intensity_agri_beach", "beach_orientation_angle"]
        
        # Initialize the internal NMF-based decomposition model.
        self.decomposition_model = MatrixDecompositionModel(config)
        
        # Placeholders for regression models to predict latent factors.
        self.time_model = None
        self.space_model = None
        
        # To store latent factor matrices from training.
        self.W = None
        self.H = None

    def train(self, data: pd.DataFrame) -> None:
        """
        Train the Matrix Decomposition Framework.
        
        This method performs:
          1. Pivoting the training data into a matrix X where rows are dates and columns are sites.
          2. Decomposing X via NMF to obtain latent factors W (temporal) and H (spatial).
          3. Training a temporal model to predict W from time features.
          4. Training a spatial model to predict H from site features.
        
        Args:
            data: Training data DataFrame containing at least the columns defined by date_column,
                  site_column, and target_column.
        """
        self.logger.info("Starting training for Matrix Decomposition Framework.")
        
        # Pivot the data: rows = unique dates, columns = unique sites, values = target measurement.
        # X_matrix = data.pivot(index=self.date_column, columns=self.site_column, values=self.target_column)
        # X_matrix = X_matrix.fillna(0)  # Fill missing values as appropriate

        data_preprocessor = Preprocessor()

        self.training_enterococci_matrix = self.decomposition_model.target_matrix(data)
        
        # Perform NMF decomposition on X_matrix.
        self.W, self.H = self.decomposition_model.decompose(self.training_enterococci_matrix)
        self.logger.info("Decomposition complete: latent factors W and H obtained.")
        
        # Prepare training data for the temporal model.
        self.time_features = [col for col in data.columns if col not in self.site_features + [self.target_column] + ['tidal_state', 'wind_shore_3h', 'wind_shore_6h', 'wind_shore_12h']]

        time_data = data.drop_duplicates(self.date_column)
        temporal_X = time_data[self.time_features]
        temporal_X = data_preprocessor.fill_missing_values(temporal_X, 'mean')
        temporal_y = pd.DataFrame(self.W, index=self.training_enterococci_matrix.index)

        temporal_X_datetime, temporal_X[self.date_column] = temporal_X[self.date_column], 0
        temporal_X = data_preprocessor.label_encode(temporal_X)

        temporal_features_debug = temporal_X.copy()
        temporal_features_debug["DateTime"] = temporal_X_datetime
        temporal_features_debug.to_csv("data/processed/temporal_features.csv", index=False)

        temporal_X = pd.read_csv("/Users/asif/Documents/Forecasting Microbial Contamination (Master's Project)/Code/Cleaned-Machine-Learning-Pipeline-Safeswim/ml_pipeline/Experiment Results/Matrix Framework/temporal_data_old.csv")
        temporal_y = pd.read_csv("/Users/asif/Documents/Forecasting Microbial Contamination (Master's Project)/Code/Cleaned-Machine-Learning-Pipeline-Safeswim/ml_pipeline/Experiment Results/Matrix Framework/temporal_y_old.csv")
        print(temporal_X)
        print(temporal_y)

        # Train the temporal multi-output regression model
        self.temporal_model = RandomForestRegressor()
        self.temporal_model.fit(temporal_X, temporal_y)
        self.logger.info("Temporal model training complete.")
        
        # Prepare training data for the spatial model.
        # Assume site_features exist and each unique site appears once.
        site_data = data.drop_duplicates(self.site_column)
        spatial_X = site_data[self.site_features]
        spatial_y = pd.DataFrame(self.H.T, index=self.training_enterococci_matrix.columns)
        spatial_X = data_preprocessor.label_encode(spatial_X)
        spatial_X["SITE_NAME"] = 0
        
        self.spatial_model = RandomForestRegressor()
        # print(spatial_X)
        # print(spatial_y)
        # spatial_X.info()
        # spatial_y.info()
        spatial_X.to_csv("data/processed/spatial_features.csv")

        self.spatial_model.fit(spatial_X, spatial_y)
        self.logger.info("Spatial model training complete.")
        
        self.logger.info("Matrix Decomposition Framework training complete.")


    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict water quality values using the Matrix Decomposition Framework.
        
        This method:
        1. Extracts unique time and site data from the input.
        2. Prepares features for the temporal and spatial models.
        3. Uses the temporal model to predict latent temporal factors (Ŵ) for each unique date.
        4. Uses the spatial model to predict latent spatial factors (Ĥ) for each unique site.
        5. Reconstructs the forecasted water quality matrix as: X̂ = Ŵ dot (Ĥ)^T.
        6. Returns a DataFrame with dates as rows and sites as columns.
        
        Args:
            data: New input data DataFrame containing at least the date_column, site_column,
                and the auxiliary features required by the temporal and spatial models.
        
        Returns:
            A DataFrame representing the predicted water quality matrix.
        """
        self.logger.info("Starting prediction using the Matrix Decomposition Framework.")
        
        # Initialize a preprocessor instance to perform label encoding.
        preprocessor = Preprocessor()
        
        # --- Prepare Temporal Data ---
        # Assume each unique date should yield one prediction row.
        time_data = data.drop_duplicates(self.date_column).copy()
        # Select the features for the temporal model.
        temporal_X = time_data[self.time_features].copy()
        # Mimic training: set the date column to a dummy value (e.g., 0)
        temporal_X[self.date_column] = 0
        # Label encode the temporal features.
        temporal_X = preprocessor.label_encode(temporal_X)
        
        # Predict latent temporal factors (Ŵ) for each date.
        W_hat = self.temporal_model.predict(temporal_X)
        # Convert to DataFrame using the unique dates as index.
        W_hat_df = pd.DataFrame(W_hat, index=time_data[self.date_column])
        
        # --- Prepare Spatial Data ---
        # Assume each unique site should yield one prediction row.
        site_data = data.drop_duplicates(self.site_column).copy()
        # Select the features for the spatial model.
        spatial_X = site_data[self.site_features].copy()
        # Label encode the spatial features.
        spatial_X = preprocessor.label_encode(spatial_X)
        
        # Predict latent spatial factors (Ĥ) for each site.
        H_hat = self.spatial_model.predict(spatial_X)
        # Convert to DataFrame using the site names as index.
        H_hat_df = pd.DataFrame(H_hat, index=site_data[self.site_column])
        
        # --- Reconstruct Predicted Matrix ---
        # Reconstruct the forecasted water quality matrix:
        # X̂ = Ŵ dot (Ĥ)^T.
        X_hat = np.dot(W_hat, H_hat.T)
        
        # Create a DataFrame with index as dates and columns as site names.
        pred_matrix = pd.DataFrame(X_hat, index=W_hat_df.index, columns=H_hat_df.index)


        # Multi-output model
        pred_matrix.reset_index(drop=False, inplace=True)

        y_pred = pred_matrix.melt(
            id_vars=['DateTime'],
            var_name='SITE_NAME',
            value_name='predictions'
            )

        y_temp = pd.DataFrame(data = {"DateTime": data["DateTime"], "SITE_NAME": data["SITE_NAME"]})

        y_pred = pd.merge(y_pred, y_temp, on=['DateTime', 'SITE_NAME'], how='right')
        y_pred.drop(columns=['SITE_NAME', 'DateTime'], inplace=True)
        y_pred = pd.Series(y_pred['predictions'], name='predictions')
        
        self.logger.info("Prediction complete using Matrix Decomposition Framework.")
        return y_pred


    def save(self, output_path: Path) -> None:
        """
        Save the Matrix Decomposition Framework instance to disk.
        
        Args:
            output_path: Path where the model should be saved.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_path)
        self.logger.info(f"Matrix Decomposition Framework saved to {output_path}")
