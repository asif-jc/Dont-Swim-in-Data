from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

def create_interactive_forecast_plot(
    test_data: pd.DataFrame, 
    train_data: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
    date_column: str = "DateTime",
    site_column: str = "SITE_NAME",
    target_column: str = "Enterococci"
) -> go.Figure:
    """Create an interactive forecast comparison plot with multiple models.
    
    Args:
        test_data: DataFrame containing test predictions and true values
        train_data: Optional DataFrame containing training predictions
        output_path: Path to save the HTML plot
        date_column: Name of the column containing dates
        site_column: Name of the column containing site names
        target_column: Name of the column containing true Enterococci values
        
    Returns:
        Plotly figure object
    """
    # Ensure date column is datetime
    for df in [test_data, train_data]:
        if df is not None and date_column in df.columns:
            if not pd.api.types.is_datetime64_dtype(df[date_column]):
                df[date_column] = pd.to_datetime(df[date_column])
    
    # Create figure with two subplots if we have both test and train data
    if train_data is not None:
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=('Test Forecast', 'Training Forecast'),
                           vertical_spacing=0.12)
    else:
        fig = make_subplots(rows=1, cols=1, 
                           subplot_titles=('Forecast',),
                           vertical_spacing=0.12)
    
    # Define color dictionary for consistent model colors
    colors = {
        'lightgbm_predictions': "#c75a93",  # Pink/purple
        'linear_regression_predictions': "#ca7040",  # Orange/brown
        'probabilistic_framework_predictions': "#5ba966",  # Green
        'matrix_decomposition_predictions': "#8176cc",  # Purple
        'decision_tree_predictions': "#cc7676",  # Red
        'mlp_predictions': "#6587cd",  # Blue
        target_column: "red",  # Ground truth always red
        'Exceedance Threshold': "red",
        'Precautionary Threshold': "orange",
    }
    
    # Get all prediction columns (ending with "_predictions")
    prediction_columns = []
    for col in test_data.columns:
        if col.endswith('_predictions'):
            model_name = col.replace('_predictions', '').replace('_', ' ').title()
            # Add tuple of (column_name, color, display_name)
            prediction_columns.append((col, colors.get(col, "gray"), model_name))
    
    # Add sorted sites for dropdown
    sites = sorted(test_data[site_column].unique())
    
    # Calculate traces per site for visibility toggling
    # +3 for ground truth and two threshold lines, *2 for train and test
    rows_count = 2 if train_data is not None else 1
    traces_per_site = (len(prediction_columns) + 3) * rows_count
    
    # Create dropdown buttons for site selection
    buttons = []
    
    for site_idx, site in enumerate(sites):
        # Process data for test and train
        test_site_data = test_data[test_data[site_column] == site].sort_values(date_column)
        train_site_data = None if train_data is None else train_data[train_data[site_column] == site].sort_values(date_column)
        
        # Cap values at 1000 for better visualization
        for dataset in [test_site_data, train_site_data]:
            if dataset is not None:
                for col in dataset.columns:
                    if col not in [date_column, site_column] and pd.api.types.is_numeric_dtype(dataset[col]):
                        dataset[col] = dataset[col].clip(upper=1000)
        
        # Create sequential x values (ignoring actual dates, as requested)
        test_x_values = list(range(len(test_site_data)))
        train_x_values = None if train_site_data is None else list(range(len(train_site_data)))
        
        # Get datetime strings for hover info
        test_datetime_str = test_site_data[date_column].dt.strftime('%Y-%m-%d %H:%M:%S')
        train_datetime_str = None if train_site_data is None else train_site_data[date_column].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Create a list of datasets to plot
        datasets = [(test_site_data, test_x_values, test_datetime_str)]
        if train_site_data is not None:
            datasets.append((train_site_data, train_x_values, train_datetime_str))
        
        # Add traces for each dataset (test and potentially train)
        for row, (data, x_vals, datetime_str) in enumerate(datasets, 1):
            name_suffix = " (Training)" if row == 2 else ""
            
            # Add prediction lines for each model
            for col, color, name in prediction_columns:
                if col in data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=data[col],
                            name=name,
                            legendgroup=name,  # Link traces between plots
                            showlegend=(row == 1),  # Only show in legend once
                            line=dict(
                                width=3,
                                color=color,
                                dash='solid'
                            ),
                            visible=(site_idx == 0),
                            customdata=datetime_str,
                            hovertemplate=f"DateTime: %{{customdata}}<br>{name}: %{{y:.2f}}<br><extra></extra>"
                        ),
                        row=row, col=1
                    )
            
            # Add ground truth (Enterococci) in red
            if target_column in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=data[target_column].clip(upper=1000),
                        name='Ground Truth',
                        legendgroup='Ground Truth',
                        showlegend=(row == 1),
                        line=dict(color='red', width=3),
                        visible=(site_idx == 0),
                        customdata=datetime_str,
                        hovertemplate=f"DateTime: %{{customdata}}<br>Ground Truth: %{{y:.2f}}<br><extra></extra>"
                    ),
                    row=row, col=1
                )
            
            # Add threshold lines
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=[280] * len(x_vals),
                    name='Exceedance Threshold',
                    legendgroup='Exceedance Threshold',
                    showlegend=(row == 1),
                    line=dict(color='red', width=2, dash='dot'),
                    visible=(site_idx == 0)
                ),
                row=row, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=[140] * len(x_vals),
                    name='Precautionary Threshold',
                    legendgroup='Precautionary Threshold',
                    showlegend=(row == 1),
                    line=dict(color='orange', width=2, dash='dot'),
                    visible=(site_idx == 0)
                ),
                row=row, col=1
            )
        
        # Set up visibility array for dropdown functionality
        visible = [False] * (len(sites) * traces_per_site)
        for i in range(traces_per_site):
            visible[site_idx * traces_per_site + i] = True
            
        buttons.append(
            dict(
                label=site,
                method="update",
                args=[{"visible": visible},
                      {"title": f"FORECAST - {site}"}]
            )
        )

    # Update layout
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.1,
                y=1.15,
                name="Site Selection"
            )
        ],
        title=f"FORECAST - {sites[0] if sites else ''}",
        height=800,
        width=1850,
        hovermode='x unified',
        showlegend=True
    )
    
    # Update axes labels and ranges
    rows_count = 2 if train_data is not None else 1
    for i in range(1, rows_count + 1):
        fig.update_xaxes(title_text="Sampling Time", row=i, col=1)
        fig.update_yaxes(title_text="Enterococci (MPN/100mL)", range=[0, 1000], row=i, col=1)
    
    # Save the figure if output path is provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        logger.info(f"Interactive forecast plot saved to {output_path}")
    
    return fig