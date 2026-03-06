"""
Scoring script for Azure ML managed endpoint
This script is loaded by Azure ML to handle inference requests
"""

import os
import json
import numpy as np
import joblib
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init():
    """
    Initialize model when the endpoint starts.
    This function is called once when the endpoint is deployed.
    """
    global model

    try:
        # Get the model path from environment variable
        model_path = os.getenv('AZUREML_MODEL_DIR', './models')

        # Find the model file
        import glob
        model_files = glob.glob(os.path.join(model_path, '**/*.pkl'), recursive=True)

        if not model_files:
            model_files = glob.glob(os.path.join(model_path, '**/*.joblib'), recursive=True)

        if not model_files:
            raise FileNotFoundError(f"No model files found in {model_path}")

        # Load the first model found (or you can specify which one)
        model_file = model_files[0]
        logger.info(f"Loading model from: {model_file}")

        model = joblib.load(model_file)
        logger.info("Model loaded successfully")

        # Log model information
        if hasattr(model, 'coef_'):
            logger.info(f"Model coefficients shape: {model.coef_.shape}")
            logger.info(f"Model intercept: {model.intercept_:.4f}")

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)
        raise


def run(raw_data: str) -> str:
    """
    Make predictions on input data.

    Args:
        raw_data: JSON string containing input data

    Returns:
        JSON string with predictions

    Expected input format:
        {
            "data": [[feature_0, feature_1, ..., feature_9], [...]],
            "metadata": {...}  # optional
        }

    Or simplified format:
        {
            "features": [[feature_0, feature_1, ..., feature_9], [...]]
        }

    Or just an array:
        [[feature_0, feature_1, ..., feature_9], [...]]
    """
    try:
        logger.info(f"Received request")

        # Parse input data
        input_data = json.loads(raw_data)

        # Handle different input formats
        if isinstance(input_data, dict):
            if 'data' in input_data:
                features = input_data['data']
            elif 'features' in input_data:
                features = input_data['features']
            else:
                # Assume all values are features
                features = [list(input_data.values())]
        elif isinstance(input_data, list):
            features = input_data
        else:
            raise ValueError("Unsupported input format. Expected dict or list.")

        # Convert to numpy array
        X = np.array(features)
        logger.info(f"Input shape: {X.shape}")

        # Make predictions
        predictions = model.predict(X)
        logger.info(f"Generated {len(predictions)} prediction(s)")

        # Prepare response
        response = {
            'predictions': predictions.tolist(),
            'model_type': type(model).__name__,
            'num_samples': len(predictions)
        }

        # Add model metadata if available
        if hasattr(model, 'coef_'):
            response['model_info'] = {
                'num_features': len(model.coef_),
                'has_intercept': model.fit_intercept
            }

        return json.dumps(response)

    except Exception as e:
        error_response = {
            'error': str(e),
            'error_type': type(e).__name__
        }
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return json.dumps(error_response)


# For local testing
if __name__ == "__main__":
    # Test the scoring script locally
    import sys

    print("Testing scoring script locally...")

    # Initialize
    print("Initializing model...")
    init()

    # Test prediction
    test_data = {
        "features": [
            [0.714, -1.515, -0.223, 0.617, -0.015, -0.366, 0.071, 1.095, -0.073, 0.626],
            [-0.234, 0.543, -0.234, -0.047, 0.028, 1.365, -0.457, 1.004, 0.767, 0.497]
        ]
    }

    print("\nInput data:")
    print(json.dumps(test_data, indent=2))

    print("\nMaking prediction...")
    result = run(json.dumps(test_data))

    print("\nPrediction result:")
    print(json.dumps(json.loads(result), indent=2))

    print("\nScoring script test completed!")
