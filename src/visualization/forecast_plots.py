from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
import sys

project_root = Path(__file__).resolve().parent.parent  
sys.path.append(str(project_root))

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

def create_forecast_visualizations(data: pd.DataFrame, output_dir: Path, 
                                config: Optional[Dict[str, Any]] = None) -> List[Path]:
    """Create visualization plots for forecast data.
    
    Args:
        data: DataFrame containing predictions and optionally true values
        output_dir: Directory to save visualizations
        config: Optional configuration dictionary
        
    Returns:
        List of paths to created visualization files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default configuration
    if config is None:
        config = {}
    
    date_col = config.get("date_column", "DateTime")
    site_col = config.get("site_column", "SITE_NAME")
    target_col = config.get("target_column", "Enterococci")
    pred_col = config.get("prediction_column", "predictions")
    
    # Ensure date column is datetime
    if pd.api.types.is_datetime64_dtype(data[date_col]):
        data = data.copy()
    else:
        data = data.copy()
        data[date_col] = pd.to_datetime(data[date_col])
    
    # Sort data by date
    data = data.sort_values(by=[date_col])
    
    # List to store created file paths
    created_files = []
    
    # 1. Create time-series plots by site
    if site_col in data.columns:
        sites = data[site_col].unique()
        
        for site in sites:
            site_data = data[data[site_col] == site]
            
            if len(site_data) < 2:
                logger.warning(f"Not enough data points for site {site}, skipping visualization")
                continue
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot actual values if available
            if target_col in site_data.columns:
                ax.plot(site_data[date_col], site_data[target_col], 'o-', label='Actual', color='blue')
            
            # Plot predictions
            ax.plot(site_data[date_col], site_data[pred_col], 's--', label='Predicted', color='red')
            
            # Add exceedance threshold line
            ax.axhline(y=280, color='r', linestyle=':', label='Exceedance Threshold (280)')
            
            # Add precautionary threshold line
            ax.axhline(y=140, color='orange', linestyle=':', label='Precautionary Threshold (140)')
            
            # Format the plot
            ax.set_title(f'Enterococci Forecast for {site}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Enterococci (MPN/100mL)')
            ax.legend()
            
            # Format date axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            fig.autofmt_xdate()
            
            # Set y-axis to log scale for better visibility
            ax.set_yscale('log')
            ax.set_ylim(bottom=1)  # Avoid log(0) issues
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Save the plot
            site_file = output_dir / f"forecast_{site.replace(' ', '_')}.png"
            plt.tight_layout()
            plt.savefig(site_file, dpi=300)
            plt.close()
            
            created_files.append(site_file)
            logger.info(f"Created time-series visualization for {site}")
    
    # 2. Create overall performance scatter plot if actual values are available
    if target_col in data.columns:
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot scatter with log scale
        ax.scatter(data[target_col], data[pred_col], alpha=0.6)
        
        # Add perfect prediction line
        max_val = max(data[target_col].max(), data[pred_col].max())
        ax.plot([0, max_val], [0, max_val], 'k--', label='Perfect Prediction')
        
        # Add threshold lines
        ax.axhline(y=280, color='r', linestyle=':', label='Exceedance Threshold')
        ax.axvline(x=280, color='r', linestyle=':')
        
        # Format the plot
        ax.set_title('Predicted vs Actual Enterococci Concentrations')
        ax.set_xlabel('Actual Enterococci (MPN/100mL)')
        ax.set_ylabel('Predicted Enterococci (MPN/100mL)')
        ax.legend()
        
        # Set log scale for both axes
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1, max_val * 1.1)
        ax.set_ylim(1, max_val * 1.1)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Save the plot
        scatter_file = output_dir / "prediction_scatter.png"
        plt.tight_layout()
        plt.savefig(scatter_file, dpi=300)
        plt.close()
        
        created_files.append(scatter_file)
        logger.info(f"Created prediction scatter plot")
    
    # 3. Create error histogram if actual values are available
    if target_col in data.columns:
        # Calculate absolute and percentage errors
        data['abs_error'] = np.abs(data[target_col] - data[pred_col])
        data['pct_error'] = 100 * data['abs_error'] / data[target_col]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Absolute error histogram
        sns.histplot(data['abs_error'], kde=True, ax=ax1)
        ax1.set_title('Absolute Error Distribution')
        ax1.set_xlabel('Absolute Error (MPN/100mL)')
        ax1.set_ylabel('Frequency')
        
        # Percentage error histogram
        sns.histplot(data['pct_error'].clip(0, 200), kde=True, ax=ax2)  # Clip extreme values
        ax2.set_title('Percentage Error Distribution')
        ax2.set_xlabel('Percentage Error (%)')
        ax2.set_ylabel('Frequency')
        
        # Save the plot
        error_file = output_dir / "error_distribution.png"
        plt.tight_layout()
        plt.savefig(error_file, dpi=300)
        plt.close()
        
        created_files.append(error_file)
        logger.info(f"Created error distribution plots")
    
    return created_files