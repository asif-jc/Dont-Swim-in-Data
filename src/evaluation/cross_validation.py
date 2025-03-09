from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import pandas as pd

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

class TimeSeriesCV:
    """Time-Series Cross-Validation for water quality forecasting.
    
    Implements a nested time-series cross-validation approach with sliding windows
    for both inner (validation) and outer (evaluation) loops.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Time-Series Cross-Validation.
        
        Args:
            config: Dictionary containing CV configuration
        """
        self.config = config
        
        # Extract test periods from config
        self.test_periods = config.get("test_periods", [
            "2021-10-01_2022-10-01",
            "2022-10-01_2023-10-01",
            "2023-10-01_2024-10-01"
        ])
        
        # Parse test periods into datetime objects
        self.parsed_periods = []
        for period in self.test_periods:
            start_str, end_str = period.split("_")
            start_date = datetime.strptime(start_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_str, "%Y-%m-%d")
            self.parsed_periods.append((start_date, end_date))
            
        logger.info(f"Initialized Time-Series CV with {len(self.test_periods)} periods")
        
    def split(self, data: pd.DataFrame, date_column: str = "sampling_date") -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate train-test splits according to the defined periods.
        
        Args:
            data: DataFrame containing the data
            date_column: Name of the column containing the dates
            
        Returns:
            List of (train, test) DataFrame tuples
        """
        # Ensure date column is datetime
        if pd.api.types.is_datetime64_dtype(data[date_column]):
            dates = data[date_column]
        else:
            dates = pd.to_datetime(data[date_column])
        
        splits = []
        
        # List of sites to remove from the test set
        sites_to_remove = [
            'Scarborough Beach by clock tower', 
            'Sumner Beach Surf club', 
            'Caroline Bay - mid beach', 
            'Timaru Coast Caroline Bay at Virtue Avenue',
            'Timaru Coast at yacht club jetty', 
            'Taylors Mistake Beach Surf club'
        ]
        
        for i, (start_date, end_date) in enumerate(self.parsed_periods):
            # Test set: data within the current period
            test_mask = (dates >= start_date) & (dates < end_date)
            test_data = data[test_mask].copy()
            # Remove unwanted sites from the test set
            test_data = test_data[~test_data["SITE_NAME"].isin(sites_to_remove)]
            
            # Training set: data before the current period
            train_mask = dates < start_date
            train_data = data[train_mask].copy()
            
            logger.info(f"Split {i+1}: Train size={len(train_data)}, Test size={len(test_data)}")
            splits.append((train_data, test_data))
            
        return splits
    
    def nested_split(self, data: pd.DataFrame, date_column: str = "sampling_date") -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """Generate nested train-validation-test splits.
        
        Args:
            data: DataFrame containing the data
            date_column: Name of the column containing the dates
            
        Returns:
            List of (train, validation, test) DataFrame tuples
        """
        # Ensure date column is datetime
        if pd.api.types.is_datetime64_dtype(data[date_column]):
            dates = data[date_column]
        else:
            dates = pd.to_datetime(data[date_column])
        
        nested_splits = []
        
        for i, (start_date, end_date) in enumerate(self.parsed_periods):
            # Test set: data within the current period
            test_mask = (dates >= start_date) & (dates < end_date)
            test_data = data[test_mask].copy()
            
            # Validation set: data from the preceding year
            val_start_date = start_date.replace(year=start_date.year - 1)
            val_end_date = start_date
            
            val_mask = (dates >= val_start_date) & (dates < val_end_date)
            val_data = data[val_mask].copy()
            
            # Training set: data before the validation period
            train_mask = dates < val_start_date
            train_data = data[train_mask].copy()
            
            logger.info(f"Nested Split {i+1}: Train size={len(train_data)}, "
                      f"Validation size={len(val_data)}, Test size={len(test_data)}")
            
            nested_splits.append((train_data, val_data, test_data))
            
        return nested_splits