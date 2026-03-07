"""
Local inference script for Linear Regression model
Supports multiple input formats and prediction modes
"""

import os
import sys
import json
import logging
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Union, List, Dict, Any, Optional

from config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelInference:
    """Handles model loading and inference operations"""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize inference handler

        Args:
            model_path: Path to saved model. If None, loads latest model.
        """
        self.config = get_config()
        self.model = None
        self.model_path = model_path
        self.feature_names = None

        self.load_model(model_path)

    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load trained model from disk

        Args:
            model_path: Path to model file. If None, loads latest model.
        """
        if model_path is None:
            # Find latest model
            models_dir = Path(self.config['models_path'])
            if not models_dir.exists():
                raise FileNotFoundError(
                    f"Models directory not found: {models_dir}\n"
                    "Please train a model first using 'make local' or 'python app.py'"
                )

            model_files = list(models_dir.glob('*.pkl')) + list(models_dir.glob('*.joblib'))
            if not model_files:
                raise FileNotFoundError(
                    f"No model files found in {models_dir}\n"
                    "Please train a model first using 'make local' or 'python app.py'"
                )

            # Get latest model by modification time
            model_path = max(model_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Loading latest model: {model_path}")
        else:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            logger.info(f"Loading model: {model_path}")

        try:
            self.model = joblib.load(model_path)
            self.model_path = model_path
            logger.info(f"Model loaded successfully")

            # Log model info
            if hasattr(self.model, 'coef_'):
                logger.info(f"Model coefficients shape: {self.model.coef_.shape}")
                logger.info(f"Model intercept: {self.model.intercept_:.4f}")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on input data

        Args:
            X: Input features array

        Returns:
            Array of predictions
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Cannot make predictions.")

        try:
            predictions = self.model.predict(X)
            logger.info(f"Generated {len(predictions)} prediction(s)")
            return predictions
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            raise

    def predict_single(self, features: List[float]) -> float:
        """
        Make prediction on a single sample

        Args:
            features: List of feature values

        Returns:
            Prediction value
        """
        X = np.array([features])
        prediction = self.predict(X)
        return float(prediction[0])

    def predict_from_csv(
        self,
        csv_path: str,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Make predictions on data from CSV file

        Args:
            csv_path: Path to CSV file with features
            output_path: Optional path to save predictions

        Returns:
            DataFrame with predictions
        """
        logger.info(f"Reading CSV file: {csv_path}")
        df = pd.read_csv(csv_path)

        # Remove target column if present
        if 'target' in df.columns:
            logger.info("Target column found - excluding from features")
            y_true = df['target'].values
            X = df.drop(columns=['target']).values
        else:
            y_true = None
            X = df.values

        # Make predictions
        predictions = self.predict(X)

        # Create results DataFrame
        results_df = df.copy()
        results_df['prediction'] = predictions

        if y_true is not None:
            results_df['target'] = y_true
            results_df['error'] = y_true - predictions
            results_df['abs_error'] = np.abs(y_true - predictions)

            # Calculate metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            mse = mean_squared_error(y_true, predictions)
            mae = mean_absolute_error(y_true, predictions)
            r2 = r2_score(y_true, predictions)

            logger.info("=" * 50)
            logger.info("PREDICTION METRICS")
            logger.info("=" * 50)
            logger.info(f"MSE: {mse:.4f}")
            logger.info(f"RMSE: {np.sqrt(mse):.4f}")
            logger.info(f"MAE: {mae:.4f}")
            logger.info(f"R² Score: {r2:.4f}")
            logger.info("=" * 50)

        # Save predictions if output path specified
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(output_path, index=False)
            logger.info(f"Predictions saved to: {output_path}")

        return results_df

    def predict_from_json(
        self,
        json_data: Union[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Make predictions from JSON data

        Args:
            json_data: JSON string or dictionary with features

        Returns:
            Dictionary with predictions
        """
        if isinstance(json_data, str):
            # Parse JSON string
            if Path(json_data).exists():
                # Read from file
                with open(json_data, 'r') as f:
                    data = json.load(f)
            else:
                # Parse JSON string
                data = json.loads(json_data)
        else:
            data = json_data

        # Handle different JSON formats
        if 'features' in data:
            # Format: {"features": [[1, 2, 3], [4, 5, 6]]}
            X = np.array(data['features'])
        elif isinstance(data, list):
            # Format: [[1, 2, 3], [4, 5, 6]]
            X = np.array(data)
        elif isinstance(data, dict) and all(isinstance(v, (int, float)) for v in data.values()):
            # Format: {"feature_0": 1, "feature_1": 2}
            X = np.array([list(data.values())])
        else:
            raise ValueError("Unsupported JSON format. Expected 'features' array or list of values.")

        predictions = self.predict(X)

        result = {
            'predictions': predictions.tolist(),
            'num_samples': len(predictions),
            'timestamp': datetime.now().isoformat()
        }

        return result


def interactive_mode(inference: ModelInference) -> None:
    """
    Interactive mode for making predictions

    Args:
        inference: ModelInference instance
    """
    logger.info("\n" + "=" * 60)
    logger.info("INTERACTIVE PREDICTION MODE")
    logger.info("=" * 60)

    # Get number of features
    if hasattr(inference.model, 'coef_'):
        n_features = len(inference.model.coef_)
    else:
        raise ValueError("Unable to determine number of features from model")

    logger.info(f"Model expects {n_features} features")
    logger.info("Enter 'q' to quit\n")

    while True:
        try:
            print(f"\nEnter {n_features} feature values (comma-separated):")
            user_input = input("> ").strip()

            if user_input.lower() == 'q':
                logger.info("Exiting interactive mode...")
                break

            # Parse input
            features = [float(x.strip()) for x in user_input.split(',')]

            if len(features) != n_features:
                print(f"Error: Expected {n_features} features, got {len(features)}")
                continue

            # Make prediction
            prediction = inference.predict_single(features)

            print(f"\n{'=' * 50}")
            print(f"Input: {features}")
            print(f"Prediction: {prediction:.4f}")
            print(f"{'=' * 50}")

        except ValueError as e:
            print(f"Error: Invalid input. Please enter numeric values. ({str(e)})")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


def main():
    """Main entry point for inference script"""
    parser = argparse.ArgumentParser(
        description='Local inference for Linear Regression model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default)
  python inference.py

  # Predict from CSV file
  python inference.py --csv data/test_data.csv

  # Predict from CSV and save results
  python inference.py --csv data/test_data.csv --output predictions.csv

  # Predict from JSON file
  python inference.py --json data/input.json

  # Predict with specific model
  python inference.py --model models/my_model.pkl --csv data/test.csv

  # Predict single instance from command line
  python inference.py --features 1.2,3.4,5.6,7.8,9.0
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        help='Path to model file (default: latest model)'
    )
    parser.add_argument(
        '--csv',
        type=str,
        help='Path to CSV file with features for batch prediction'
    )
    parser.add_argument(
        '--json',
        type=str,
        help='Path to JSON file or JSON string with features'
    )
    parser.add_argument(
        '--features',
        type=str,
        help='Comma-separated feature values for single prediction'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output path for saving predictions'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )

    args = parser.parse_args()

    try:
        # Initialize inference
        inference = ModelInference(model_path=args.model)

        # Determine mode
        if args.csv:
            # CSV prediction mode
            logger.info("Running CSV prediction mode...")
            results = inference.predict_from_csv(args.csv, args.output)

            # Display sample results
            print("\n" + "=" * 60)
            print("PREDICTION RESULTS (first 10 rows)")
            print("=" * 60)
            print(results.head(10).to_string(index=False))
            print("=" * 60)
            print(f"\nTotal predictions: {len(results)}")

            if args.output:
                print(f"Full results saved to: {args.output}")

        elif args.json:
            # JSON prediction mode
            logger.info("Running JSON prediction mode...")
            results = inference.predict_from_json(args.json)

            print("\n" + "=" * 60)
            print("PREDICTION RESULTS")
            print("=" * 60)
            print(json.dumps(results, indent=2))
            print("=" * 60)

            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Results saved to: {args.output}")

        elif args.features:
            # Single prediction from command line
            logger.info("Running single prediction mode...")
            features = [float(x.strip()) for x in args.features.split(',')]
            prediction = inference.predict_single(features)

            print("\n" + "=" * 60)
            print("PREDICTION RESULT")
            print("=" * 60)
            print(f"Input features: {features}")
            print(f"Prediction: {prediction:.4f}")
            print("=" * 60)

        else:
            # Interactive mode (default)
            interactive_mode(inference)

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
