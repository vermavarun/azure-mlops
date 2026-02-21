"""
Configuration management for MLOps pipeline
"""

import os
import json
from pathlib import Path
from typing import Dict, Any


def get_config() -> Dict[str, Any]:
    """
    Load configuration from environment variables or config file.
    
    Returns:
        Dictionary containing pipeline configuration
    """
    
    config = {
        # Azure ML Configuration
        'subscription_id': os.getenv('AZURE_SUBSCRIPTION_ID', '1f8817ba-4f67-4ce6-9b70-e4b60fd2c3db'),
        'resource_group_name': os.getenv('AZURE_RESOURCE_GROUP', 'mlops-lr-dev-rg-ab837c05'),
        'workspace_name': os.getenv('AZURE_WORKSPACE_NAME', 'mlops-lr-dev-amlws-c773a9ba'),
        
        # Compute Resources
        'compute_cluster_name': os.getenv('COMPUTE_CLUSTER', 'mlops-lr-dev-cc'),
        'compute_instance_name': os.getenv('COMPUTE_INSTANCE', 'mlops-lr-dev-ci'),
        
        # Data Configuration
        'test_size': float(os.getenv('TEST_SIZE', '0.2')),
        'random_state': int(os.getenv('RANDOM_STATE', '42')),
        'data_path': Path(os.getenv('DATA_PATH', './data')),
        
        # Model Configuration
        'model_name': os.getenv('MODEL_NAME', 'linear-regression-lr'),
        'model_version': os.getenv('MODEL_VERSION', '1'),
        'fit_intercept': os.getenv('FIT_INTERCEPT', 'true').lower() == 'true',
        'normalize': os.getenv('NORMALIZE', 'false').lower() == 'true',
        
        # Training Configuration
        'max_iter': int(os.getenv('MAX_ITER', '1000')),
        'tol': float(os.getenv('TOL', '0.0001')),
        
        # Output Configuration
        'output_path': Path(os.getenv('OUTPUT_PATH', './outputs')),
        'models_path': Path(os.getenv('MODELS_PATH', './models')),
        'metrics_path': Path(os.getenv('METRICS_PATH', './metrics')),
    }
    
    return config


def save_config(config: Dict[str, Any], path: str = 'config.json') -> None:
    """Save configuration to JSON file"""
    config_copy = config.copy()
    # Convert Path objects to strings for JSON serialization
    for key, value in config_copy.items():
        if isinstance(value, Path):
            config_copy[key] = str(value)
    
    with open(path, 'w') as f:
        json.dump(config_copy, f, indent=2)


def load_config_from_file(path: str = 'config.json') -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(path, 'r') as f:
        config = json.load(f)
    
    # Convert string paths back to Path objects
    path_keys = ['data_path', 'output_path', 'models_path', 'metrics_path']
    for key in path_keys:
        if key in config:
            config[key] = Path(config[key])
    
    return config
