"""
Demo script showing various pipeline usage patterns
"""

import logging
from pathlib import Path

from config import get_config
from data_handler import load_data, save_data
from train import train_model, save_model
from evaluate import (
    evaluate_model,
    evaluate_cross_validation,
    plot_predictions,
    plot_residual_distribution,
    save_metrics
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_basic_workflow():
    """Demonstrate basic training and evaluation workflow"""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 1: Basic Workflow (Linear Regression)")
    logger.info("=" * 60)
    
    config = get_config()
    
    # Load data
    X_train, X_test, y_train, y_test = load_data(
        test_size=config['test_size'],
        random_state=config['random_state']
    )
    
    # Train model
    model = train_model(
        X_train, y_train,
        model_type='linear',
        fit_intercept=config['fit_intercept']
    )
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save outputs
    save_model(model, output_path='models', model_name='linear_regression_basic')
    save_metrics(metrics, output_path='metrics/metrics_basic.json')
    
    # Create visualizations
    y_pred = model.predict(X_test)
    plot_predictions(y_test, y_pred, output_path='metrics/predictions_basic.png')
    plot_residual_distribution(y_test, y_pred, output_path='metrics/residuals_basic.png')
    
    logger.info(f"R² Score: {metrics['r2_score']:.4f}")
    logger.info("Demo 1 completed successfully!\n")


def demo_model_comparison():
    """Demonstrate comparing multiple model types"""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 2: Model Comparison (Linear, Ridge, Lasso)")
    logger.info("=" * 60)
    
    config = get_config()
    
    # Load data
    X_train, X_test, y_train, y_test = load_data(
        test_size=config['test_size'],
        random_state=config['random_state']
    )
    
    models_info = {
        'linear': {'type': 'linear'},
        'ridge': {'type': 'ridge', 'alpha': 1.0},
        'lasso': {'type': 'lasso', 'alpha': 0.1},
    }
    
    results = {}
    
    for model_name, model_params in models_info.items():
        logger.info(f"\nTraining {model_name} model...")
        
        # Train model
        model = train_model(X_train, y_train, **model_params)
        
        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        results[model_name] = metrics['r2_score']
        
        # Save
        save_model(model, output_path='models', model_name=f'{model_name}_comparison')
        save_metrics(metrics, output_path=f'metrics/metrics_{model_name}.json')
    
    # Print comparison
    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON RESULTS")
    logger.info("=" * 60)
    for model_name, r2_score in results.items():
        logger.info(f"{model_name:10s}: R² = {r2_score:.4f}")
    
    best_model = max(results, key=results.get)
    logger.info(f"\nBest Model: {best_model} with R² = {results[best_model]:.4f}")
    logger.info("Demo 2 completed successfully!\n")


def demo_cross_validation():
    """Demonstrate cross-validation evaluation"""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 3: Cross-Validation Evaluation")
    logger.info("=" * 60)
    
    config = get_config()
    
    # Load data (use full dataset for CV)
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = train_model(X_train, y_train, model_type='linear')
    
    # Perform cross-validation
    cv_results = evaluate_cross_validation(model, X_train, y_train, cv=5)
    
    logger.info("\n" + "=" * 60)
    logger.info("CROSS-VALIDATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"R² Score (mean):     {cv_results['r2_mean']:.4f} +/- {cv_results['r2_std']:.4f}")
    logger.info(f"RMSE (mean):         {cv_results['rmse_mean']:.4f} +/- {cv_results['rmse_std']:.4f}")
    logger.info(f"MAE (mean):          {cv_results['mae_mean']:.4f} +/- {cv_results['mae_std']:.4f}")
    
    logger.info("Demo 3 completed successfully!\n")


def demo_save_load_data():
    """Demonstrate saving and loading processed data"""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 4: Data Persistence")
    logger.info("=" * 60)
    
    config = get_config()
    
    # Load and save data
    X_train, X_test, y_train, y_test = load_data(
        test_size=config['test_size'],
        random_state=config['random_state']
    )
    
    save_data(X_train, X_test, y_train, y_test, output_path='data/splits')
    logger.info("Data saved successfully")
    
    # Load saved data
    import numpy as np
    from pathlib import Path
    
    data_dir = Path('data/splits')
    X_train_loaded = np.load(data_dir / 'X_train.npy')
    y_train_loaded = np.load(data_dir / 'y_train.npy')
    
    # Verify
    assert np.allclose(X_train, X_train_loaded), "Loaded data doesn't match"
    assert np.allclose(y_train, y_train_loaded), "Loaded target doesn't match"
    
    logger.info("Data loaded and verified successfully")
    logger.info("Demo 4 completed successfully!\n")


def demo_polynomial_regression():
    """Demonstrate polynomial regression"""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 5: Polynomial Regression")
    logger.info("=" * 60)
    
    config = get_config()
    
    # Load data
    X_train, X_test, y_train, y_test = load_data(
        test_size=config['test_size'],
        random_state=config['random_state']
    )
    
    # Train polynomial models with different degrees
    degrees = [1, 2, 3]
    results = {}
    
    for degree in degrees:
        logger.info(f"\nTraining polynomial model with degree={degree}...")
        
        poly_features, model = train_model(
            X_train, y_train,
            model_type='polynomial',
            degree=degree
        )
        
        # Transform test data
        X_test_poly = poly_features.transform(X_test)
        
        # Evaluate
        metrics = evaluate_model(model, X_test_poly, y_test)
        results[degree] = metrics['r2_score']
        
        # Save
        save_model(
            (poly_features, model),
            output_path='models',
            model_name=f'polynomial_degree{degree}'
        )
        save_metrics(metrics, output_path=f'metrics/metrics_poly_degree{degree}.json')
    
    # Print comparison
    logger.info("\n" + "=" * 60)
    logger.info("POLYNOMIAL REGRESSION COMPARISON")
    logger.info("=" * 60)
    for degree, r2_score in results.items():
        logger.info(f"Degree {degree}: R² = {r2_score:.4f}")
    
    logger.info("Demo 5 completed successfully!\n")


if __name__ == "__main__":
    # Create output directories
    Path('models').mkdir(exist_ok=True)
    Path('metrics').mkdir(exist_ok=True)
    Path('data/splits').mkdir(parents=True, exist_ok=True)
    
    # Run all demos
    try:
        demo_basic_workflow()
        demo_model_comparison()
        demo_cross_validation()
        demo_save_load_data()
        demo_polynomial_regression()
        
        logger.info("\n" + "=" * 60)
        logger.info("ALL DEMOS COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("\nOutput files:")
        logger.info("  Models:  models/")
        logger.info("  Metrics: metrics/")
        logger.info("  Data:    data/splits/")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}", exc_info=True)
