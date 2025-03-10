from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from functools import reduce
import dash
import os
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, recall_score, precision_score, 
    confusion_matrix, fbeta_score
)

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))


from src.utils.logging import setup_logger

logger = setup_logger(__name__)

# Constants
MEDIUM_THRESHOLD = 280  # For exceedance classification

def generate_dashboard(forecast):
    """
    Generate a Dash dashboard for visualizing and comparing model forecasts.
    
    Args:
        forecast (dict): Dictionary of DataFrames with model predictions
        
    Returns:
        dash.Dash: Dashboard application
    """
    logger.info("Generating dashboard...")
    
    # Define common columns from the first model's DataFrame:
    first_df = next(iter(forecast.values()))
    common_cols = [col for col in first_df.columns if col != "predictions" and not col.startswith("q_")]

    dfs = []
    for model, df in forecast.items():
        df_copy = df.copy()
        df_copy = df_copy.rename(columns={"predictions": model})
        
        if model == "probabilistic_framework":
            quantile_cols = [col for col in df_copy.columns if col.startswith("q_")]
            selected_cols = common_cols + quantile_cols + [model]
        else:
            selected_cols = common_cols + [model]
        
        dfs.append(df_copy[selected_cols])

    # Merge on the common columns
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=common_cols, how='inner'), dfs)
    data = merged_df.copy()

    # Create performance tables for each model
    model_names = list(forecast.keys())
    performance_tables = {}
    
    for model in model_names:
        temp_df = data.copy()
        temp_df["PREDICTION"] = temp_df[model]
        performance_tables[model] = create_performance_table(temp_df)
    
    # Create Dash application
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    
    # Define layout with banner image
    app.layout = html.Div([
        # Banner Image
        html.Img(
            src= r'C:\Users\AsifCheena\OneDrive - BSL\Desktop\IMPORTANT REPOS\Dont-Swim-in-Data\assets\beach.png',  # Path to the banner image
            style={'width': '100%', 'height': 'auto', 'margin-bottom': '20px'}
        ),
        
        # Dashboard Title
        html.H1(
            "FORECAST PERFORMANCE DASHBOARD",
            style={'textAlign': 'center', 'color': 'green', 'margin-bottom': '20px'}
        ),
        
        # Dropdowns for site, plot type, and model selection
        html.Div([
            html.Div([
                html.H3("Site Selection", style={'textAlign': 'center'}),
                dcc.Dropdown(
                    id='site-dropdown',
                    options=[{'label': site, 'value': site} for site in data['SITE_NAME'].unique()],
                    value=data['SITE_NAME'].unique()[0],
                    style={'width': '100%'}
                ),
            ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px'}),
            
            html.Div([
                html.H3("Plot Type", style={'textAlign': 'center'}),
                dcc.Dropdown(
                    id='plot-type-dropdown',
                    options=[
                        {'label': 'Model Comparison', 'value': 'model_comparison'},
                        {'label': 'Quantile Forecast', 'value': 'quantile_forecast'}
                    ],
                    value='model_comparison',
                    style={'width': '100%'}
                ),
            ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px'}),
            
            html.Div([
                html.H3("Model Selection", style={'textAlign': 'center'}),
                dcc.Dropdown(
                    id='model-dropdown',
                    options=[{'label': model, 'value': model} for model in model_names],
                    value=model_names[0] if model_names else None,
                    style={'width': '100%'}
                ),
            ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px'}),
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'padding': '10px'}),
        
        # Graph for forecasts
        html.Br(),
        dcc.Graph(
            id='forecast-graph',
            style={'width': '100%', 'height': '70vh'}
        ),
        
        # Performance Metrics Table
        html.Br(),
        html.H2("Performance Metrics", style={'textAlign': 'center'}),
        dash_table.DataTable(
            id='performance-table',
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'center',
                'padding': '5px'
            },
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 0},
                    'backgroundColor': 'rgb(240, 240, 240)',
                    'fontWeight': 'bold'
                }
            ]
        )
    ])
    
    # Define callback for updating the graph based on dropdowns
    @app.callback(
        Output('forecast-graph', 'figure'),
        [
            Input('site-dropdown', 'value'),
            Input('plot-type-dropdown', 'value'),
            Input('model-dropdown', 'value')
        ]
    )
    def update_graph(selected_site, selected_plot_type, selected_model):
        # Filter data for the selected site
        site_data = data[data["SITE_NAME"] == selected_site]
        
        if selected_plot_type == 'model_comparison':
            return generate_model_comparison_plot(site_data, model_names, y_max=1000)
        elif selected_plot_type == 'quantile_forecast':
            lower_quantile = next((col for col in site_data.columns if col.startswith('q_0.0') or col.startswith('q_0.1')), None)
            upper_quantile = next((col for col in site_data.columns if col.startswith('q_0.9')), None)
            
            return generate_quantile_forecast_plot(
                site_data, 
                selected_site, 
                selected_model,
                lower_bound_col=lower_quantile,
                upper_bound_col=upper_quantile,
                cap_value=1000
            )
        
        return go.Figure()
    
    # Define callback for updating the performance table based on model selection
    @app.callback(
        Output('performance-table', 'data'),
        Output('performance-table', 'columns'),
        [Input('model-dropdown', 'value')]
    )
    def update_performance_table(selected_model):
        if selected_model and selected_model in performance_tables:
            df = performance_tables[selected_model]
            columns = [{"name": i, "id": i} for i in df.columns]
            return df.to_dict('records'), columns
        
        return [], []
    
    return app

def generate_model_comparison_plot(site_data, model_names, y_max=1000):
    """
    Generate a plot comparing different model predictions for a specific site.
    
    Args:
        site_data (DataFrame): DataFrame containing site-specific data
        model_names (list): List of model names to include in the plot
        y_max (int): Maximum value for y-axis
        
    Returns:
        plotly.graph_objects.Figure: Comparison plot
    """
    fig = go.Figure()
    
    # Sort data by datetime to ensure correct ordering
    site_data = site_data.sort_values("DateTime")
    
    # Create evenly spaced x-values (0, 1, 2, ...) while retaining datetime labels
    x_values = list(range(len(site_data)))
    datetime_labels = site_data["DateTime"].dt.strftime('%Y-%m-%d %H:%M')
    
    # Cap values at y_max for better visualization
    for col in site_data.columns:
        if col not in ["DateTime", "SITE_NAME"] and pd.api.types.is_numeric_dtype(site_data[col]):
            site_data[col] = site_data[col].clip(upper=y_max)
    
    # Add ground truth (Enterococci)
    if "Enterococci" in site_data.columns:
        fig.add_trace(go.Scatter(
            x=x_values, 
            y=site_data["Enterococci"], 
            mode='lines', 
            name='Actual Values',
            line=dict(color='red', width=4),
            hovertext=datetime_labels
        ))
    
    # Add model predictions
    colors = [
        "#5ba966",  # Green
        "#6587cd",  # Blue
        "#c75a93",  # Pink/purple
        "#ca7040",  # Orange/brown
        "#8176cc",  # Purple
        "#cc7676",  # Red
    ]
    
    # Add each model's prediction
    for i, model in enumerate(model_names):
        if model in site_data.columns:
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=x_values, 
                y=site_data[model], 
                mode='lines', 
                name=model,
                line=dict(color=color, width=2),
                hovertext=datetime_labels
            ))
    
    # Add threshold lines
    fig.add_trace(go.Scatter(
        x=x_values,
        y=[280] * len(site_data),
        mode='lines',
        name='Exceedance Threshold',
        line=dict(color='red', dash='dot', width=1.5),
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=x_values,
        y=[140] * len(site_data),
        mode='lines',
        name='Precautionary Threshold',
        line=dict(color='orange', dash='dot', width=1.5),
        hoverinfo='skip'
    ))
    
    # Update layout with customized x-axis
    fig.update_layout(
        title={
            'text': f"Model Comparison - {site_data['SITE_NAME'].iloc[0]}",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis=dict(
            title="Sampling Time",
            tickmode='array',
            tickvals=x_values,
            ticktext=datetime_labels,
            tickangle=45,
            # Only show a subset of ticks if there are many points
            nticks=min(20, len(x_values))
        ),
        yaxis_title="Enterococci (MPN/100mL)",
        yaxis=dict(range=[0, y_max]),
        legend_title="Models",
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

def generate_quantile_forecast_plot(site_data, site_name, selected_model, lower_bound_col=None, upper_bound_col=None, cap_value=1000):
    """
    Generate a quantile forecast plot for a specific model and site.
    
    Args:
        site_data (DataFrame): DataFrame containing site-specific data
        site_name (str): Name of the site
        selected_model (str): Name of the selected model
        lower_bound_col (str): Column name for lower bound quantile
        upper_bound_col (str): Column name for upper bound quantile
        cap_value (int): Maximum value for y-axis
        
    Returns:
        plotly.graph_objects.Figure: Quantile forecast plot
    """
    fig = go.Figure()
    
    # Sort data by datetime to ensure correct ordering
    site_data = site_data.sort_values("DateTime")
    
    # Create evenly spaced x-values (0, 1, 2, ...) while retaining datetime labels
    x_values = list(range(len(site_data)))
    datetime_labels = site_data["DateTime"].dt.strftime('%Y-%m-%d %H:%M')
    
    # Cap values for better visualization
    for col in site_data.columns:
        if col not in ["DateTime", "SITE_NAME"] and pd.api.types.is_numeric_dtype(site_data[col]):
            site_data[col] = site_data[col].clip(upper=cap_value)
    
    # Add actual values
    if "Enterococci" in site_data.columns:
        fig.add_trace(go.Scatter(
            x=x_values, 
            y=site_data["Enterococci"], 
            mode='lines', 
            name='Actual Values',
            line=dict(color='red', width=4),
            hovertext=datetime_labels
        ))
    
    # Add model prediction
    if selected_model in site_data.columns:
        fig.add_trace(go.Scatter(
            x=x_values, 
            y=site_data[selected_model], 
            mode='lines', 
            name=f'{selected_model} Prediction',
            line=dict(color='green', width=3),
            hovertext=datetime_labels
        ))
    
    # Add quantile bounds if available
    if lower_bound_col and upper_bound_col and lower_bound_col in site_data.columns and upper_bound_col in site_data.columns:
        # Create the fill between upper and lower bounds
        fig.add_trace(go.Scatter(
            x=x_values + x_values[::-1],
            y=site_data[upper_bound_col].tolist() + site_data[lower_bound_col][::-1].tolist(),
            fill='toself',
            fillcolor='rgba(0, 255, 0, 0.2)',  # Semi-transparent green
            line=dict(color='rgba(255,255,255,0)'),
            name='Prediction Interval',
            showlegend=True,
            hoverinfo='skip'
        ))
    
    # Add threshold lines
    fig.add_trace(go.Scatter(
        x=x_values,
        y=[280] * len(site_data),
        mode='lines',
        name='Exceedance Threshold',
        line=dict(color='red', dash='dot', width=1.5),
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=x_values,
        y=[140] * len(site_data),
        mode='lines',
        name='Precautionary Threshold',
        line=dict(color='orange', dash='dot', width=1.5),
        hoverinfo='skip'
    ))
    
    # Update layout with customized x-axis
    fig.update_layout(
        title={
            'text': f"{selected_model} Forecast with Uncertainty - {site_name}",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis=dict(
            title="Sampling Time",
            tickmode='array',
            tickvals=x_values,
            ticktext=datetime_labels,
            tickangle=45,
            # Only show a subset of ticks if there are many points
            nticks=min(20, len(x_values))
        ),
        yaxis_title="Enterococci (MPN/100mL)",
        yaxis=dict(range=[0, cap_value]),
        legend_title="Legend",
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

def create_performance_table(data):
    """
    Create a performance metrics table for a specific model.
    
    Args:
        data (DataFrame): DataFrame containing actual values and predictions
        
    Returns:
        DataFrame: Performance metrics table
    """
    metrics_keys = [
        "rmse", "mae", "mape", "nrmse", "r2",
        "mae_exceedance", "mape_exceedance", "nrmse_exceedance",
        "mae_safe", "mape_safe", "nrmse_safe",
        "accuracy", "recall_safe", "recall_exceedance", "precision_safe", "precision_exceedance",
        "sensitivity", "specificity", "tn", "fp", "fn", "tp", 
        "fbeta_safe_1_5", "fbeta_exceedance_1_5", "fbeta_safe_2", "fbeta_exceedance_2", "precautionary", 
        "recall_precautionary", "weighted_mape", "weighted_mape_safe", "weighted_mape_exceedance", "r2_exceedance", "r2_safe"
    ]
    
    performance_df = pd.DataFrame(columns=[
        'SITE_NAME', 'Accuracy', 'Sensitivity/Recall (EXCEEDANCE)', 'Specificity',
        'TP', 'FN', 'TN', 'FP', 'Log-Weighted MAPE',
        'Log-Weighted MAPE (EXCEEDANCE)', 'Log-Weighted MAPE (SAFE)'
    ])
    
    # Calculate overall performance
    test_metrics = {key: [] for key in metrics_keys}
    
    if 'Enterococci' in data.columns and 'PREDICTION' in data.columns:
        y_true = convert_target_to_flag(data['Enterococci'], 280)
        y_pred = convert_target_to_flag(data['PREDICTION'], 280)
        
        # Calculate performance metrics
        performance_classification_metrics(
            y_true, y_pred,
            data['Enterococci'],
            data['PREDICTION'],
            test_metrics
        )
        
        performance_regression_metrics(
            data['Enterococci'],
            data['PREDICTION'],
            test_metrics
        )
        
        # Add overall row
        overall_row = pd.DataFrame({
            'SITE_NAME': ['OVERALL'],
            'Accuracy': [round(test_metrics['accuracy'][-1], 2)],
            'Sensitivity/Recall (EXCEEDANCE)': [round(test_metrics['recall_exceedance'][-1], 2)],
            'Specificity': [round(test_metrics['specificity'][-1], 2)],
            'TP': [test_metrics['tp'][-1]],
            'FN': [test_metrics['fn'][-1]],
            'TN': [test_metrics['tn'][-1]],
            'FP': [test_metrics['fp'][-1]],
            'Log-Weighted MAPE': [round(test_metrics['mape'][-1], 2)],
            'Log-Weighted MAPE (EXCEEDANCE)': [round(test_metrics['mape_exceedance'][-1], 2)],
            'Log-Weighted MAPE (SAFE)': [round(test_metrics['mape_safe'][-1], 2)]
        })
        
        performance_df = pd.concat([performance_df, overall_row])
        
        # Calculate site-specific performance
        for site in data['SITE_NAME'].unique():
            # Reset metrics for each site
            site_metrics = {key: [] for key in metrics_keys}
            
            site_data = data[data['SITE_NAME'] == site]
            
            y_true = convert_target_to_flag(site_data['Enterococci'], 280)
            y_pred = convert_target_to_flag(site_data['PREDICTION'], 280)
            
            performance_classification_metrics(
                y_true, y_pred,
                site_data['Enterococci'],
                site_data['PREDICTION'],
                site_metrics
            )
            
            performance_regression_metrics(
                site_data['Enterococci'],
                site_data['PREDICTION'],
                site_metrics
            )
            
            site_row = pd.DataFrame({
                'SITE_NAME': [site],
                'Accuracy': [round(site_metrics['accuracy'][-1], 2)],
                'Sensitivity/Recall (EXCEEDANCE)': [round(site_metrics['recall_exceedance'][-1], 2)],
                'Specificity': [round(site_metrics['specificity'][-1], 2)],
                'TP': [site_metrics['tp'][-1]],
                'FN': [site_metrics['fn'][-1]],
                'TN': [site_metrics['tn'][-1]],
                'FP': [site_metrics['fp'][-1]],
                'Log-Weighted MAPE': [round(site_metrics['mape'][-1], 2)],
                'Log-Weighted MAPE (EXCEEDANCE)': [round(site_metrics['mape_exceedance'][-1], 2)],
                'Log-Weighted MAPE (SAFE)': [round(site_metrics['mape_safe'][-1], 2)]
            })
            
            performance_df = pd.concat([performance_df, site_row])
    
    return performance_df

def convert_target_to_flag(y, threshold):
    """Convert numeric values to 'EXCEEDANCE' or 'SAFE' based on threshold"""
    return pd.Series(["EXCEEDANCE" if val >= threshold else "SAFE" for val in y])

def performance_regression_metrics(y_true, y_pred, metrics):
    """Calculate regression performance metrics"""
    def log_mape(y_true, y_pred):
        """Calculate Mean Absolute Percentage Error on log-transformed data"""
        y_true_log = np.log1p(y_true)
        y_pred_log = np.log1p(y_pred)
        
        # Avoid division by zero
        mask = y_true_log != 0
        y_true_log = y_true_log[mask]
        y_pred_log = y_pred_log[mask]
        
        if len(y_true_log) == 0:
            return -1
        
        log_APE = np.abs((y_true_log - y_pred_log) / y_true_log)
        log_MAPE = np.mean(log_APE)
        return log_MAPE
    
    def weighted_mape(y_true, y_pred):
        """Calculate Weighted Mean Absolute Percentage Error"""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        
        # Avoid division by zero
        mask = y_true != 0
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            return -1
        
        # Calculate weights (normalized true values)
        weights = y_true / np.sum(y_true)
        
        # Calculate individual absolute percentage errors
        absolute_percentage_errors = np.abs((y_true - y_pred) / y_true)
        
        # Calculate weighted average
        wmape = np.sum(weights * absolute_percentage_errors)
        
        return wmape
    
    # Handle empty series
    if len(y_true) == 0:
        y_true = pd.Series([5])
    if len(y_pred) == 0:
        y_pred = pd.Series([5])
    
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Replace non-positive values with 5
    y_pred[y_pred <= 0] = 5
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))
    mae = mean_absolute_error(y_true, y_pred)
    wmape = weighted_mape(y_true, y_pred)
    mape = weighted_mape(np.log1p(y_true), np.log1p(y_pred))
    r2 = r2_score(np.log1p(y_true), np.log1p(y_pred))
    
    # Calculate NRMSE as a percentage
    range_y_true = np.max(np.log1p(y_true)) - np.min(np.log1p(y_true))
    nrmse = (rmse / range_y_true) * 100 if range_y_true != 0 else -1
    
    # Separate exceedance and safe cases
    exceedance_indices = y_true >= MEDIUM_THRESHOLD
    safe_indices = y_true < MEDIUM_THRESHOLD
    
    y_exceedance = y_true[exceedance_indices]
    y_pred_exceedance = y_pred[exceedance_indices]
    y_safe = y_true[safe_indices]
    y_pred_safe = y_pred[safe_indices]
    
    # Calculate metrics for exceedance cases
    if len(y_exceedance) == 0:
        mae_exceedance = -1
        mape_exceedance = -1
        nrmse_exceedance = -1
        wmape_exceedance = -1
        r2_exceedance = -1
    else:
        mae_exceedance = mean_absolute_error(y_exceedance, y_pred_exceedance)
        wmape_exceedance = weighted_mape(y_exceedance, y_pred_exceedance)
        mape_exceedance = weighted_mape(np.log1p(y_exceedance), np.log1p(y_pred_exceedance))
        
        range_y_exceedance = np.max(y_exceedance) - np.min(y_exceedance)
        nrmse_exceedance = (np.sqrt(mean_squared_error(y_exceedance, y_pred_exceedance)) / range_y_exceedance) * 100 if range_y_exceedance != 0 else -1
        
        r2_exceedance = r2_score(np.log1p(y_exceedance), np.log1p(y_pred_exceedance))
    
    # Calculate metrics for safe cases
    if len(y_safe) == 0:
        mae_safe = -1
        mape_safe = -1
        nrmse_safe = -1
        wmape_safe = -1
        r2_safe = -1
    else:
        mae_safe = mean_absolute_error(y_safe, y_pred_safe)
        wmape_safe = weighted_mape(y_safe, y_pred_safe)
        mape_safe = weighted_mape(np.log1p(y_safe), np.log1p(y_pred_safe))
        
        range_y_safe = np.max(y_safe) - np.min(y_safe)
        nrmse_safe = (np.sqrt(mean_squared_error(y_safe, y_pred_safe)) / range_y_safe) * 100 if range_y_safe != 0 else -1
        
        r2_safe = r2_score(np.log1p(y_safe), np.log1p(y_pred_safe))
    
    # Update metrics dictionary
    metrics["rmse"].append(rmse)
    metrics["mae"].append(mae)
    metrics["mape"].append(mape)
    metrics["nrmse"].append(nrmse)
    metrics["mae_exceedance"].append(mae_exceedance)
    metrics["mape_exceedance"].append(mape_exceedance)
    metrics["nrmse_exceedance"].append(nrmse_exceedance)
    metrics["mae_safe"].append(mae_safe)
    metrics["mape_safe"].append(mape_safe)
    metrics["nrmse_safe"].append(nrmse_safe)
    metrics["weighted_mape"].append(wmape)
    metrics["weighted_mape_safe"].append(wmape_safe)
    metrics["weighted_mape_exceedance"].append(wmape_exceedance)
    metrics["r2"].append(r2)
    metrics["r2_exceedance"].append(r2_exceedance)
    metrics["r2_safe"].append(r2_safe)
    
    return metrics


def performance_classification_metrics(y_true, y_pred, y_true_regression, y_pred_regression, metrics):
    """Calculate classification performance metrics for model evaluation
    
    Args:
        y_true: Series of true labels ('SAFE' or 'EXCEEDANCE')
        y_pred: Series of predicted labels ('SAFE' or 'EXCEEDANCE')
        y_true_regression: Series of true numeric values (Enterococci counts)
        y_pred_regression: Series of predicted numeric values
        metrics: Dictionary to store calculated metrics
        
    Returns:
        Dictionary with updated metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate recall scores
    recall_safe = recall_score(y_true, y_pred, pos_label="SAFE", zero_division=0)
    recall_exceedance = recall_score(y_true, y_pred, pos_label="EXCEEDANCE", zero_division=0)
    
    # Calculate precision scores
    precision_safe = precision_score(y_true, y_pred, pos_label="SAFE", zero_division=0)
    precision_exceedance = precision_score(y_true, y_pred, pos_label="EXCEEDANCE", zero_division=0)
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=["SAFE", "EXCEEDANCE"])
    
    # Handle case where confusion matrix is 1x1 (only one class present)
    if conf_matrix.shape == (1, 1):
        if y_true.iloc[0] == "SAFE":
            # Only SAFE cases
            tn, fp, fn, tp = conf_matrix[0, 0], 0, 0, 0
        else:
            # Only EXCEEDANCE cases
            tn, fp, fn, tp = 0, 0, 0, conf_matrix[0, 0]
    else:
        # Normal case with both classes
        tn, fp, fn, tp = conf_matrix.ravel()
    
    # Calculate sensitivity and specificity
    sensitivity = (recall_exceedance + recall_safe) / 2
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Calculate F-beta scores
    fbeta_safe_1_5 = fbeta_score(y_true, y_pred, pos_label="SAFE", beta=1.5, zero_division=0)
    fbeta_exceedance_1_5 = fbeta_score(y_true, y_pred, pos_label="EXCEEDANCE", beta=1.5, zero_division=0)
    fbeta_safe_2 = fbeta_score(y_true, y_pred, pos_label="SAFE", beta=2, zero_division=0)
    fbeta_exceedance_2 = fbeta_score(y_true, y_pred, pos_label="EXCEEDANCE", beta=2, zero_division=0)
    
    # Calculate precautionary metrics
    precautionary_cases = sum((y_true == "EXCEEDANCE") & (y_pred_regression >= 140) & (y_pred_regression < 280))
    all_actual_exceedances = sum(y_true == "EXCEEDANCE")
    recall_precautionary = (precautionary_cases + tp) / all_actual_exceedances if all_actual_exceedances > 0 else 0
    
    # Store metrics
    metrics["accuracy"].append(accuracy)
    metrics["recall_safe"].append(recall_safe)
    metrics["recall_exceedance"].append(recall_exceedance)
    metrics["precision_safe"].append(precision_safe)
    metrics["precision_exceedance"].append(precision_exceedance)
    metrics["sensitivity"].append(sensitivity)
    metrics["specificity"].append(specificity)
    metrics["tn"].append(tn)
    metrics["fp"].append(fp)
    metrics["fn"].append(fn)
    metrics["tp"].append(tp)
    metrics["fbeta_safe_1_5"].append(fbeta_safe_1_5)
    metrics["fbeta_exceedance_1_5"].append(fbeta_exceedance_1_5)
    metrics["fbeta_safe_2"].append(fbeta_safe_2)
    metrics["fbeta_exceedance_2"].append(fbeta_exceedance_2)
    metrics["precautionary"].append(precautionary_cases + tp)
    metrics["recall_precautionary"].append(recall_precautionary)

    return metrics