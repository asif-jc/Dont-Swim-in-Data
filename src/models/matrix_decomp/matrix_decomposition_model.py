import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import NMF
import sys

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

from src.utils.logging import setup_logger
from src.data.preprocessing import Preprocessor

logger = setup_logger(__name__)

class MatrixDecompositionModel:
    """
    Implements Non-negative Matrix Factorization (NMF) for decomposing a data matrix into latent
    temporal (W) and spatial (H) factors.
    
    The number of latent factors and other NMF parameters are read from the configuration.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.n_components = config.get("n_components", 4)  # Default: 4 latent factors
        self.init = config.get("init", "nndsvd")
        self.alpha = config.get("alpha", 0.0)
        self.l1_ratio = config.get("l1_ratio", 0.0)
        self.random_state = config.get("random_state", 42)
        self.tol = config.get("tol", 1e-4)
        self.max_iter = config.get("max_iter", 1000)
        self.solver = config.get("solver", "mu")
        self.nmf = None

    def decompose(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose the input data matrix X into latent factors W and H using NMF.
        
        Args:
            X: A DataFrame with rows as samples (e.g. dates) and columns as features (e.g. sites).
        
        Returns:
            W: np.ndarray with shape (n_samples, n_components) representing temporal factors.
            H: np.ndarray with shape (n_components, n_features) representing spatial factors.
        """
        logger.info("Performing NMF decomposition on input data matrix.")
        # self.nmf = NMF(n_components=self.n_components,
        #                init=self.init,
        #                alpha_W=self.alpha,
        #                alpha_H='same',
        #                l1_ratio=self.l1_ratio,
        #                tol=self.tol,
        #                random_state=self.random_state)
        self.nmf = NMF(n_components=self.n_components,
                       init=self.init,
                       solver=self.solver,
                       max_iter=self.max_iter,
                       random_state=self.random_state)
        W = self.nmf.fit_transform(X)
        H = self.nmf.components_

        X.to_csv("data/processed/matrix_decomp_data.csv")
        pd.DataFrame(W).to_csv("data/processed/W.csv")
        pd.DataFrame(H).to_csv("data/processed/H.csv")

        logger.info(f"NMF decomposition complete: W shape {W.shape}, H shape {H.shape}")
        return W, H

    def save(self, output_path: Path) -> None:
        """
        Save the Matrix Decomposition Model to disk.
        
        Args:
            output_path: Path where the model will be saved.
        """
        logger.info(f"Saving Matrix Decomposition Model to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_path)
        logger.info("Matrix Decomposition Model saved successfully.")

    def target_matrix(self, data: pd.DataFrame) -> pd.DataFrame:

        # Preprocess the data
        preprocessor = Preprocessor()
        processed_data = preprocessor.label_encode(data)
        processed_data = preprocessor.fill_missing_values(processed_data)
        processed_data["SITE_NAME"] = data["SITE_NAME"].astype("category")

        # Filter out specific sites
        ### .... ###


        single_target_data = processed_data.copy()
        single_target_data = single_target_data.drop_duplicates(subset=['SITE_NAME', 'DateTime'], keep='first')
        # Pivot the data to create multi-target format
        multi_target_data = single_target_data.pivot(index='DateTime', columns='SITE_NAME', values='Enterococci')
        multi_target_data = multi_target_data.reset_index()
        unimputed_data = multi_target_data.copy()

        daily_filled_matrix = preprocessor.fill_daily_measurements(multi_target_data)
        filled_matrix = preprocessor.complete_matrix_with_row_means(daily_filled_matrix)

        return filled_matrix

    @classmethod
    def load(cls, input_path: Path) -> "MatrixDecompositionModel":
        """
        Load a previously saved Matrix Decomposition Model from disk.
        
        Args:
            input_path: Path to the saved model file.
            
        Returns:
            An instance of MatrixDecompositionModel.
        """
        model = joblib.load(input_path)
        return model
