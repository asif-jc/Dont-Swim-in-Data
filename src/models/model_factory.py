from typing import Dict, Any

from src.models.probabilistic.main_probabilistic import ProbabilisticForecastingModel
from src.models.matrix_decomp.main_matrix import MatrixDecompositionFramework
# from src.models.benchmarks.linear_models import LinearRegression
# from src.models.benchmarks.tree_models import DecisionTree
from src.models.benchmarks.lightgbm_models import LightGBMModel

class ModelFactory:
    """Factory class for creating model instances."""
    
    def __init__(self, model_config: Dict[str, Any]):
        """Initialize the model factory.
        
        Args:
            model_config: Configuration dictionary for all models
        """
        self.model_config = model_config
        
    def get_model(self, model_name: str):
        """Get an instance of the specified model.
        
        Args:
            model_name: Name of the model to initialize
            
        Returns:
            An initialized model instance
        """
        if model_name == "probabilistic_framework":
            return ProbabilisticForecastingModel(self.model_config)
            
        elif model_name == "matrix_decomposition_framework":
            return MatrixDecompositionFramework(self.model_config)
            
        elif model_name == "linear_regression":
            # return LinearRegression(self.model_config["benchmarks"][model_name])
            pass
            
        elif model_name == "decision_tree":
            # return DecisionTree(self.model_config["benchmarks"][model_name])
            pass
            
        elif model_name == "lightgbm":
            return LightGBMModel(self.model_config)
            
        else:
            raise ValueError(f"Unknown model: {model_name}")