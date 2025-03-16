from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
import sys
from sklearn.preprocessing import LabelEncoder

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

class Preprocessor:

    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path) if config_path else {}

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:

        DATA_SUBSET = ['DateTime', 'SITE_NAME', 'Harbour', 'Enterococci', '3H', '6H', '12H', '24H', '48H', '72H', 'Season', 'tidal_state', 'Shallowness', 
                    'Soil_type', 'Catchment_slope', 'Landcover_catchment', 'watercraft_use', 'sewage_discharge_beach', 'high_intensity_agri_beach', 'rain_intensity_48h', 
                    'rain_duration_48h', 'wind_speed_12h', 'wind_speed_6h', 'wind_speed_3h', 'hours_to_high_tide', 'high_tide_height', 
                    'wind_direction_3h', 'wind_direction_6h', 'wind_direction_12h', 'beach_orientation_angle', 'Latitude', 'Longitude'] 
        
        data = data[DATA_SUBSET]

        data['DateTime'] = pd.to_datetime(data['DateTime'], format='%m/%d/%y %H:%M')
        data['DateTime'] = data['DateTime'].dt.strftime('%d/%m/%Y %H:%M')
        data['DateTime'] = pd.to_datetime(data['DateTime'], format='%d/%m/%Y %H:%M')

        data.sort_values(by=['SITE_NAME', 'DateTime'], inplace=True)
        data = data.drop_duplicates(subset=['DateTime', 'SITE_NAME'], keep='first')
        data.reset_index(drop=True, inplace=True)

        return data
    
    def transform_catergorical_variable_type(self, df):
        CAT_FEATURES = ["SITE_NAME", "WEEK", "DAY_OF_WEEK", "MONTH", "YEAR", "Season", "Harbour", "tidal_state", 'Shallowness', 'Soil_type', 
                        'Landcover_catchment', 'watercraft_use', 'high_intensity_agri_beach', 'potential_animal_contamination', 'stormwater_outlet_beach', 
                        'sewage_discharge_beach', 'Timing', 'WEEKEND', 'Landcover_site', 'ENSO', 'TIME_OF_DAY', 'wind_shore_3h', 'wind_shore_6h', 'wind_shore_12h']
        for c in CAT_FEATURES:
            if c in df.columns:
                df[c] = df[c].astype('category')
        return df
    
    def label_encode(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Label encode specified categorical columns. If columns is None, all columns
        with 'category' dtype will be encoded.
        """
        df = df.copy()
        if columns is None:
            # Use columns already converted to category
            columns = df.select_dtypes(include=['category', 'object']).columns.tolist()
        
        for col in columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        return df

    def fill_missing_values(self, df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
        """
        Fill missing values in numeric columns using mean or median imputation.
        
        Args:
            df: Input DataFrame.
            strategy: 'mean' or 'median' to specify imputation strategy.
            
        Returns:
            DataFrame with missing numeric values imputed.
        """
        df = df.copy()
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            else:
                raise ValueError("Strategy must be 'mean' or 'median'")
        return df
    
    def fill_daily_measurements(self, multi_target_data):
        """
        Fill measurements within each day for sites that have measurements that day.
        """
        # Make sure DateTime is the index
        if 'DateTime' in multi_target_data.columns:
            multi_target_data = multi_target_data.set_index('DateTime')
        
        # Create copy to work with
        filled_matrix = multi_target_data.copy()
        
        # Convert index to datetime
        filled_matrix.index = pd.to_datetime(filled_matrix.index)
        
        # Get unique dates
        dates = filled_matrix.index.strftime('%Y-%m-%d').unique()
        
        # For each date
        for date in dates:
            # Get rows for this date
            mask = filled_matrix.index.strftime('%Y-%m-%d') == date
            day_data = filled_matrix[mask]
            
            # For each site that has at least one measurement this day
            for column in filled_matrix.columns:
                valid_data = day_data[column].dropna()
                if len(valid_data) > 0:
                    # Get first valid measurement for this site today
                    value = valid_data.iloc[0]
                    # Fill it to other timestamps this day for this site
                    filled_matrix.loc[mask, column] = value
        
        return filled_matrix
    
    def complete_matrix_with_row_means(self, filled_matrix):
        """
        Fill any remaining NaN values in the matrix with the mean of each row.
        If a row is all NaN, it will be filled with the global mean.
        
        Parameters:
        -----------
        filled_matrix : pd.DataFrame
            DataFrame with DateTime as index and sites as columns
            
        Returns:
        --------
        pd.DataFrame
            Completed matrix with no NaN values
        """
        # Create a copy to work with
        completed_matrix = filled_matrix.copy()
        
        # Calculate row means (mean across sites for each timestamp)
        row_means = completed_matrix.mean(axis=1)
        
        # Fill remaining NaN values with the row mean
        for column in completed_matrix.columns:
            # For each column (site), fill NaN with the row mean
            mask = completed_matrix[column].isna()
            completed_matrix.loc[mask, column] = row_means[mask]
            
        return completed_matrix
    
    def set_max_target_value(self, data: pd.DataFrame) -> pd.DataFrame:
        MAX_ENTEROCOCCI_VALUE = 10000
        data = data.clip(upper=MAX_ENTEROCOCCI_VALUE)
        return data

    def _load_config(self, config_path: Path) -> Dict:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Dictionary containing configuration parameters
        """
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
        


# data = Preprocessor()

# df = data.clean_data(TRAINING_DATA_PATH)

# print(df.head())
# print(df.info())