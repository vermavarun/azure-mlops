"""
Model evaluation module for linear regression pipeline
Computes various performance metrics
"""

import logging
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate model performance on test set
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target values
        
    Returns:
        Dictionary containing evaluation metrics
    """
    
    logger.info("Evaluating model on test set...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2_score': r2_score(y_test, y_pred),
        'mape': mean_absolute_percentage_error(y_test, y_pred),
    }
    
    # Calculate additional metrics
    residuals = y_test - y_pred
    metrics['residuals_mean'] = float(np.mean(residuals))
    metrics['residuals_std'] = float(np.std(residuals))
    metrics['max_residual'] = float(np.max(np.abs(residuals)))
    
    logger.info("=" * 60)
    logger.info("MODEL EVALUATION METRICS")
    logger.info("=" * 60)
    logger.info(f"Mean Squared Error (MSE):        {metrics['mse']:.6f}")
    logger.info(f"Root Mean Squared Error (RMSE):  {metrics['rmse']:.6f}")
    logger.info(f"Mean Absolute Error (MAE):       {metrics['mae']:.6f}")
    logger.info(f"Mean Absolute % Error (MAPE):    {metrics['mape']:.6f}")
    logger.info(f"R² Score:                        {metrics['r2_score']:.6f}")
    logger.info(f"Residuals Mean:                  {metrics['residuals_mean']:.6f}")
    logger.info(f"Residuals Std:                   {metrics['residuals_std']:.6f}")
    logger.info(f"Max Residual:                    {metrics['max_residual']:.6f}")
    logger.info("=" * 60)
    
    return metrics


def evaluate_cross_validation(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5
) -> Dict[str, Any]:
    """
    Perform cross-validation evaluation
    
    Args:
        model: Model or pipeline
        X: Features
        y: Target values
        cv: Number of cross-validation folds
        
    Returns:
        Dictionary containing CV results
    """
    from sklearn.model_selection import cross_validate
    
    logger.info(f"Running {cv}-fold cross-validation...")
    
    # Define scoring metrics
    scoring = {
        'r2': 'r2',
        'neg_mse': 'neg_mean_squared_error',
        'neg_mae': 'neg_mean_absolute_error'
    }
    
    # Perform cross-validation
    cv_results = cross_validate(
        model,
        X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True
    )
    
    # Summarize results
    results = {
        'r2_mean': np.mean(cv_results['test_r2']),
        'r2_std': np.std(cv_results['test_r2']),
        'rmse_mean': np.sqrt(-np.mean(cv_results['test_neg_mse'])),
        'rmse_std': np.sqrt(np.std(cv_results['test_neg_mse'])),
        'mae_mean': -np.mean(cv_results['test_neg_mae']),
        'mae_std': np.std(cv_results['test_neg_mae']),
    }
    
    logger.info(f"Cross-validation R² Score: {results['r2_mean']:.6f} (+/- {results['r2_std']:.6f})")
    logger.info(f"Cross-validation RMSE: {results['rmse_mean']:.6f} (+/- {results['rmse_std']:.6f})")
    
    return results


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str = './metrics/predictions.png'
) -> None:
    """
    Plot actual vs predicted values
    
    Args:
        y_true: True values
        y_pred: Predicted values
        output_path: Path to save plot
    """
    
    logger.info(f"Plotting predictions to: {output_path}")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Actual vs Predicted plot
    axes[0].scatter(y_true, y_pred, alpha=0.6)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0].set_xlabel('True Values')
    axes[0].set_ylabel('Predicted Values')
    axes[0].set_title('Actual vs Predicted')
    axes[0].grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.6)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()
    
    logger.info("Plot saved successfully")


def plot_residual_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str = './metrics/residuals.png'
) -> None:
    """
    Plot residual distribution
    
    Args:
        y_true: True values
        y_pred: Predicted values
        output_path: Path to save plot
    """
    
    logger.info(f"Plotting residual distribution to: {output_path}")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram of residuals
    axes[0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Residuals')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Residual Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # Q-Q plot for normality check
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()
    
    logger.info("Residual plot saved successfully")


def save_metrics(
    metrics: Dict[str, float],
    output_path: str = './metrics/metrics.json'
) -> None:
    """
    Save metrics to JSON file
    
    Args:
        metrics: Dictionary of metrics
        output_path: Path to save metrics
    """
    
    logger.info(f"Saving metrics to: {output_path}")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    metrics_json = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics_json[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            metrics_json[key] = float(value)
        else:
            metrics_json[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    logger.info("Metrics saved successfully")


def load_metrics(metrics_path: str) -> Dict[str, float]:
    """
    Load metrics from JSON file
    
    Args:
        metrics_path: Path to metrics file
        
    Returns:
        Dictionary of metrics
    """
    
    logger.info(f"Loading metrics from: {metrics_path}")
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    logger.info("Metrics loaded successfully")
    
    return metrics
