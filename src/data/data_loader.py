from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
import sys

project_root = Path(__file__).resolve().parent.parent.parent  
sys.path.append(str(project_root))

from src.config.paths import (
    # ENTEROCOCCI_DATA_PATH,
    # METEOROLOGICAL_DATA_PATH,
    # SITE_METADATA_PATH,
    TRAINING_DATA_PATH
)

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

class DataLoader:

    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path) if config_path else {}


    def load_processed_data(self, data_path: Path) -> pd.DataFrame:
        path = Path(self.config.get('training_data_path', TRAINING_DATA_PATH))
        logger.info(f"Loading partially processed data from {path}")
        
        try:
            df = pd.read_csv(path)
            logger.info(f"Loaded {len(df)} rows and {len(df.columns)} features")
            return df
        except Exception as e:
            logger.error(f"Failed to load processed data: {e}")
            raise


    def _load_config(self, config_path: Path) -> Dict:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Dictionary containing configuration parameters
        """
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
        