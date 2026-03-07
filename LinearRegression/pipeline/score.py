"""
Scoring script for Azure ML managed endpoint
This script is loaded by Azure ML to handle inference requests
"""

import os
import json
import numpy as np
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
model = None


def init():
    """
    Initialize model when the endpoint starts.
    This function is called once when the endpoint is deployed.
    """
    global model

    try:
        # Get the model path from environment variable
        model_path = os.getenv('AZUREML_MODEL_DIR', './models')
        
        logger.info(f"AZUREML_MODEL_DIR: {model_path}")
        logger.info(f"Model directory exists: {os.path.exists(model_path)}")

        # List contents of model directory
        if os.path.exists(model_path):
            logger.info(f"Contents of {model_path}: {os.listdir(model_path)}")
        
        # Find and load the model file
        model_file = None
        
        # First try to find .pkl files
        for file in os.listdir(model_path):
            if file.endswith('.pkl'):
                model_file = os.path.join(model_path, file)
                logger.info(f"Found .pkl file: {model_file}")
                break
        
        # If no .pkl, try .joblib
        if not model_file:
            for file in os.listdir(model_path):
                if file.endswith('.joblib'):
                    model_file = os.path.join(model_path, file)
                    logger.info(f"Found .joblib file: {model_file}")
                    break
        
        if not model_file:
            raise FileNotFoundError(f"No model file found in {model_path}")

        logger.info(f"Loading model from: {model_file}")
        model = joblib.load(model_file)
        logger.info("Model loaded successfully")

        # Log model information
        if hasattr(model, 'coef_'):
            logger.info(f"Model coefficients shape: {model.coef_.shape}")
        if hasattr(model, 'intercept_'):
            logger.info(f"Model intercept: {model.intercept_:.4f}")

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)
        raise Exception(f"Failed to load model: {str(e)}")


def run(raw_data):
    """
    Make predictions on input data.

    Args:
        raw_data: JSON string or bytes containing input data

    Returns:
        JSON string with predictions

    Expected input formats:
        {"data": [[feature_0, feature_1, ..., feature_9], [...]]}
        {"features": [[feature_0, feature_1, ..., feature_9], [...]]}
        [[feature_0, feature_1, ..., feature_9], [...]]
    """
    try:
        logger.info(f"Received request")
        
        # Handle both string and bytes input
        if isinstance(raw_data, bytes):
            raw_data = raw_data.decode('utf-8')
        
        # Parse input data
        input_data = json.loads(raw_data)
        logger.info(f"Input data type: {type(input_data)}")

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
            raise ValueError(f"Unsupported input format: {type(input_data)}")

        # Convert to numpy array
        X = np.asarray(features, dtype=np.float64)
        logger.info(f"Input shape: {X.shape}")

        # Make predictions
        predictions = model.predict(X)
        logger.info(f"Predictions: {predictions}")

        # Format output
        if len(predictions) == 1:
            output = {
                "predictions": float(predictions[0])
            }
        else:
            output = {
                "predictions": predictions.tolist()
            }

        return json.dumps(output)

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        return json.dumps({
            "error": str(e),
            "predictions": None
        })
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
