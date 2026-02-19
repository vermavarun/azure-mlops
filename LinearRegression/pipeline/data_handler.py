"""
Data handling module for linear regression pipeline
Handles data loading, preprocessing, and preparation
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression

logger = logging.getLogger(__name__)


def generate_synthetic_data(
    n_samples: int = 100,
    n_features: int = 10,
    noise: float = 0.1,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic regression dataset for demonstration
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        noise: Standard deviation of noise
        random_state: Random seed
        
    Returns:
        DataFrame with features and target
    """
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state
    )
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    logger.info(f"Generated synthetic dataset: {df.shape[0]} samples, {n_features} features")
    
    return df


def load_data_from_csv(
    path: str,
    target_column: str = 'target'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data from CSV file
    
    Args:
        path: Path to CSV file
        target_column: Name of target column
        
    Returns:
        Tuple of features (X) and target (y)
    """
    logger.info(f"Loading data from: {path}")
    
    df = pd.read_csv(path)
    
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values
    
    logger.info(f"Loaded data shape: X={X.shape}, y={y.shape}")
    
    return X, y


def load_data(
    test_size: float = 0.2,
    random_state: int = 42,
    use_synthetic: bool = True,
    data_path: str = None,
    scaling: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and split data into train and test sets
    
    Args:
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        use_synthetic: Whether to use synthetic data or load from file
        data_path: Path to data file (if not using synthetic)
        scaling: Whether to apply StandardScaling
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    
    # Load data
    if use_synthetic:
        logger.info("Using synthetic dataset")
        df = generate_synthetic_data(random_state=random_state)
        X = df.drop(columns=['target']).values
        y = df['target'].values
    else:
        if data_path is None:
            raise ValueError("data_path must be provided when use_synthetic=False")
        X, y = load_data_from_csv(data_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )
    
    logger.info(f"Data split - Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # Scale features
    if scaling:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        logger.info("Features scaled using StandardScaler")
    
    return X_train, X_test, y_train, y_test


def prepare_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    remove_outliers: bool = False,
    outlier_threshold: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data by handling outliers and validating data quality
    
    Args:
        X_train: Training features
        X_test: Test features
        remove_outliers: Whether to remove outliers
        outlier_threshold: Number of standard deviations for outlier detection
        
    Returns:
        Tuple of processed (X_train, X_test)
    """
    
    if remove_outliers:
        logger.info(f"Removing outliers (threshold: {outlier_threshold} std)")
        
        # Detect outliers based on Z-score
        from scipy import stats
        z_scores = np.abs(stats.zscore(X_train))
        mask = (z_scores < outlier_threshold).all(axis=1)
        X_train = X_train[mask]
        
        logger.info(f"Removed {(~mask).sum()} outliers. New size: {X_train.shape[0]}")
    
    # Validate data
    assert not np.any(np.isnan(X_train)), "NaN values found in training data"
    assert not np.any(np.isinf(X_train)), "Inf values found in training data"
    assert not np.any(np.isnan(X_test)), "NaN values found in test data"
    assert not np.any(np.isinf(X_test)), "Inf values found in test data"
    
    logger.info("Data validation passed")
    
    return X_train, X_test


def save_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    output_path: str = './data'
) -> None:
    """
    Save train/test split to files
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        output_path: Directory to save files
    """
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / 'X_train.npy', X_train)
    np.save(output_dir / 'X_test.npy', X_test)
    np.save(output_dir / 'y_train.npy', y_train)
    np.save(output_dir / 'y_test.npy', y_test)
    
    logger.info(f"Data saved to: {output_dir}")
