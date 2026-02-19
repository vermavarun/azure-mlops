"""
Experiment configuration templates for different scenarios.

Usage:
    from experiment_configs import get_experiment_config
    
    config = get_experiment_config('baseline')
    # ... use config for training
"""

# Baseline experiment - simple linear regression
BASELINE = {
    'name': 'baseline_linear',
    'model_type': 'linear',
    'model_kwargs': {},
    'data_config': {
        'use_synthetic': True,
        'n_samples': 100,
        'n_features': 10,
        'test_size': 0.2,
        'random_state': 42,
        'scale': True,
    },
    'training': {
        'epochs': 1,
        'batch_size': None,
    },
}

# Polynomial features experiment
POLYNOMIAL = {
    'name': 'polynomial_features',
    'model_type': 'polynomial',
    'model_kwargs': {'degree': 2},
    'data_config': {
        'use_synthetic': True,
        'n_samples': 150,
        'n_features': 5,
        'test_size': 0.2,
        'random_state': 42,
        'scale': True,
        'remove_outliers': True,
    },
    'training': {
        'epochs': 1,
        'batch_size': None,
    },
}

# Ridge regression with regularization
RIDGE = {
    'name': 'ridge_l2_reg',
    'model_type': 'ridge',
    'model_kwargs': {'alpha': 1.0},
    'data_config': {
        'use_synthetic': True,
        'n_samples': 200,
        'n_features': 15,
        'test_size': 0.2,
        'random_state': 42,
        'scale': True,
        'remove_outliers': True,
    },
    'training': {
        'epochs': 1,
        'batch_size': None,
    },
}

# Lasso regression with feature selection
LASSO = {
    'name': 'lasso_l1_reg',
    'model_type': 'lasso',
    'model_kwargs': {'alpha': 0.1},
    'data_config': {
        'use_synthetic': True,
        'n_samples': 200,
        'n_features': 20,
        'test_size': 0.2,
        'random_state': 42,
        'scale': True,
        'remove_outliers': True,
    },
    'training': {
        'epochs': 1,
        'batch_size': None,
    },
}

# Large scale experiment
LARGE_SCALE = {
    'name': 'large_scale',
    'model_type': 'ridge',
    'model_kwargs': {'alpha': 0.5},
    'data_config': {
        'use_synthetic': True,
        'n_samples': 1000,
        'n_features': 50,
        'test_size': 0.2,
        'random_state': 42,
        'scale': True,
        'remove_outliers': True,
    },
    'training': {
        'epochs': 1,
        'batch_size': None,
    },
}

# Hyperparameter tuning grid
HYPERPARAMETER_GRID = {
    'linear': [{}],
    'polynomial': [
        {'degree': 1},
        {'degree': 2},
        {'degree': 3},
    ],
    'ridge': [
        {'alpha': 0.01},
        {'alpha': 0.1},
        {'alpha': 1.0},
        {'alpha': 10.0},
    ],
    'lasso': [
        {'alpha': 0.001},
        {'alpha': 0.01},
        {'alpha': 0.1},
        {'alpha': 1.0},
    ],
}

# Model comparison experiment
COMPARISON = {
    'name': 'model_comparison',
    'models': ['linear', 'polynomial', 'ridge', 'lasso'],
    'data_config': {
        'use_synthetic': True,
        'n_samples': 200,
        'n_features': 10,
        'test_size': 0.2,
        'random_state': 42,
        'scale': True,
    },
}

# Available experiments
EXPERIMENTS = {
    'baseline': BASELINE,
    'polynomial': POLYNOMIAL,
    'ridge': RIDGE,
    'lasso': LASSO,
    'large_scale': LARGE_SCALE,
    'comparison': COMPARISON,
}


def get_experiment_config(name):
    """Get configuration for named experiment.
    
    Args:
        name: Experiment name ('baseline', 'polynomial', 'ridge', 'lasso',
              'large_scale', 'comparison')
    
    Returns:
        Dictionary with experiment configuration
        
    Raises:
        KeyError: If experiment name not found
    """
    if name not in EXPERIMENTS:
        available = ', '.join(EXPERIMENTS.keys())
        raise KeyError(f"Unknown experiment '{name}'. Available: {available}")
    
    return EXPERIMENTS[name].copy()


def list_experiments():
    """List all available experiments."""
    return list(EXPERIMENTS.keys())


if __name__ == '__main__':
    # Display all experiments
    for name, config in EXPERIMENTS.items():
        print(f"\n{'='*50}")
        print(f"Experiment: {name}")
        print(f"{'='*50}")
        for key, value in config.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key}: {sub_value}")
            else:
                print(f"{key}: {value}")
