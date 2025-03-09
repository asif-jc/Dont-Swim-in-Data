# config/paths.py
from pathlib import Path

# Determine the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Define common data directories
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "src" / "config"

# Define specific dataset paths
# METEOROLOGICAL_DATA_PATH = RAW_DATA_DIR / "meteorological_data.csv"
# ENTEROCOCCI_DATA_PATH = RAW_DATA_DIR / "enterococci_data.csv"
# SITE_METADATA_PATH = RAW_DATA_DIR / "site_metadata.csv"
MAIN_CONFIG_PATH = CONFIG_DIR / "main_config.yaml"
TRAINING_DATA_PATH = INTERIM_DATA_DIR / "training_data.csv"
