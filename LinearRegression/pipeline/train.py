"""
Training module for linear regression model
"""

import logging
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)


def train_linear_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    fit_intercept: bool = True,
    #normalize: bool = False,
    positive: bool = False
) -> LinearRegression:
    """
    Train a linear regression model

    Args:
        X_train: Training features
        y_train: Training target
        fit_intercept: Whether to fit intercept
        normalize: Whether to normalize features
        positive: If True, coefficients must be positive

    Returns:
        Trained LinearRegression model
    """

    logger.info("Training Linear Regression model...")

    # Create and train model
    model = LinearRegression(
        fit_intercept=fit_intercept,
        #normalize=normalize,
        positive=positive
    )

    model.fit(X_train, y_train)

    logger.info(f"Model trained successfully")
    logger.info(f"Intercept: {model.intercept_:.4f}")
    logger.info(f"Coefficients shape: {model.coef_.shape}")
    logger.info(f"R² score (train): {model.score(X_train, y_train):.4f}")

    return model


def train_polynomial_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    degree: int = 2,
    fit_intercept: bool = True
) -> Tuple[PolynomialFeatures, LinearRegression]:
    """
    Train a polynomial regression model

    Args:
        X_train: Training features
        y_train: Training target
        degree: Degree of polynomial features
        fit_intercept: Whether to fit intercept

    Returns:
        Tuple of (PolynomialFeatures transformer, LinearRegression model)
    """

    logger.info(f"Training Polynomial Regression model (degree={degree})...")

    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly_features.fit_transform(X_train)

    logger.info(f"Polynomial features shape: {X_train_poly.shape}")

    # Train linear regression on polynomial features
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(X_train_poly, y_train)

    logger.info(f"Polynomial model trained successfully")
    logger.info(f"R² score (train): {model.score(X_train_poly, y_train):.4f}")

    return poly_features, model


def train_ridge_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float = 1.0
):
    """
    Train a Ridge regression model

    Args:
        X_train: Training features
        y_train: Training target
        alpha: Regularization strength

    Returns:
        Trained Ridge model
    """
    from sklearn.linear_model import Ridge

    logger.info(f"Training Ridge Regression model (alpha={alpha})...")

    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    logger.info(f"Ridge model trained successfully")
    logger.info(f"R² score (train): {model.score(X_train, y_train):.4f}")

    return model


def train_lasso_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float = 0.1,
    max_iter: int = 1000
):
    """
    Train a Lasso regression model

    Args:
        X_train: Training features
        y_train: Training target
        alpha: Regularization strength
        max_iter: Maximum iterations

    Returns:
        Trained Lasso model
    """
    from sklearn.linear_model import Lasso

    logger.info(f"Training Lasso Regression model (alpha={alpha})...")

    model = Lasso(alpha=alpha, max_iter=max_iter)
    model.fit(X_train, y_train)

    logger.info(f"Lasso model trained successfully")
    logger.info(f"R² score (train): {model.score(X_train, y_train):.4f}")
    logger.info(f"Non-zero coefficients: {np.count_nonzero(model.coef_)}")

    return model


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = 'linear',
    **kwargs
):
    """
    Train a regression model

    Args:
        X_train: Training features
        y_train: Training target
        model_type: Type of model ('linear', 'polynomial', 'ridge', 'lasso')
        **kwargs: Additional arguments for model

    Returns:
        Trained model
    """

    if model_type.lower() == 'linear':
        return train_linear_regression(X_train, y_train, **kwargs)
    elif model_type.lower() == 'polynomial':
        return train_polynomial_regression(X_train, y_train, **kwargs)
    elif model_type.lower() == 'ridge':
        return train_ridge_regression(X_train, y_train, **kwargs)
    elif model_type.lower() == 'lasso':
        return train_lasso_regression(X_train, y_train, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def save_model(
    model,
    output_path: str = './models',
    model_name: str = 'linear_regression'
) -> Path:
    """
    Save trained model to disk

    Args:
        model: Trained model
        output_path: Directory to save model
        model_name: Name of model file

    Returns:
        Path to saved model
    """

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f'{model_name}.pkl'
    joblib.dump(model, model_path)

    logger.info(f"Model saved to: {model_path}")

    return model_path


def load_model(model_path: str):
    """
    Load trained model from disk

    Args:
        model_path: Path to model file

    Returns:
        Loaded model
    """

    logger.info(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

    logger.info("Model loaded successfully")

    return model
