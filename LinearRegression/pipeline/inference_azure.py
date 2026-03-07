"""
Client for making predictions using Azure ML deployed endpoint
"""

import os
import sys
import json
import logging
import argparse
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AzureMLInference:
    """Client for Azure ML endpoint inference"""

    def __init__(
        self,
        scoring_uri: str = None,
        api_key: str = None,
        endpoint_name: str = None
    ):
        """
        Initialize Azure ML inference client

        Args:
            scoring_uri: Endpoint scoring URI
            api_key: Endpoint authentication key
            endpoint_name: Endpoint name (to load config from file)
        """
        self.scoring_uri = scoring_uri
        self.api_key = api_key

        # Load from config file if not provided
        if not self.scoring_uri or not self.api_key:
            self._load_from_config(endpoint_name)

        if not self.scoring_uri or not self.api_key:
            raise ValueError(
                "Scoring URI and API key are required. "
                "Provide them directly or ensure endpoint_config.json exists."
            )

        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }

        logger.info(f"Initialized Azure ML client for: {self.scoring_uri}")

    def _load_from_config(self, endpoint_name: str = None):
        """Load endpoint configuration from file"""
        config_file = Path("outputs") / "endpoint_config.json"

        if config_file.exists():
            logger.info(f"Loading endpoint config from: {config_file}")
            with open(config_file, 'r') as f:
                config = json.load(f)

            self.scoring_uri = config.get('scoring_uri')
            self.api_key = config.get('primary_key')

            logger.info(f"Loaded config for endpoint: {config.get('endpoint_name')}")
        else:
            logger.warning(f"Config file not found: {config_file}")
            logger.info("Deploy endpoint first using: make deploy-azure")

    def predict(
        self,
        data: Union[np.ndarray, List[List[float]], Dict[str, Any]],
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Make prediction request to Azure ML endpoint

        Args:
            data: Input data (array, list, or dict)
            timeout: Request timeout in seconds

        Returns:
            Dictionary with prediction results
        """
        # Prepare request payload
        if isinstance(data, np.ndarray):
            payload = {"features": data.tolist()}
        elif isinstance(data, list):
            payload = {"features": data}
        elif isinstance(data, dict):
            payload = data
        else:
            raise ValueError("Data must be numpy array, list, or dict")

        # Make request
        logger.info(f"Sending request to: {self.scoring_uri}")

        try:
            response = requests.post(
                self.scoring_uri,
                json=payload,
                headers=self.headers,
                timeout=timeout
            )

            response.raise_for_status()
            result = response.json()

            logger.info(f"Received response with {result.get('num_samples', 0)} predictions")
            return result

        except requests.exceptions.Timeout:
            logger.error(f"Request timed out after {timeout} seconds")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            if hasattr(e.response, 'text'):
                logger.error(f"Response: {e.response.text}")
            raise

    def predict_single(self, features: List[float]) -> float:
        """
        Predict single sample

        Args:
            features: List of feature values

        Returns:
            Prediction value
        """
        result = self.predict([features])
        predictions = result.get('predictions', [])

        if not predictions:
            raise ValueError("No predictions returned from endpoint")

        return float(predictions[0])

    def predict_from_csv(
        self,
        csv_path: str,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Make predictions on data from CSV file

        Args:
            csv_path: Path to CSV file
            output_path: Optional path to save results

        Returns:
            DataFrame with predictions
        """
        logger.info(f"Reading CSV: {csv_path}")
        df = pd.read_csv(csv_path)

        # Remove target column if present
        if 'target' in df.columns:
            y_true = df['target'].values
            X = df.drop(columns=['target']).values
        else:
            y_true = None
            X = df.values

        # Make predictions
        result = self.predict(X)
        predictions = np.array(result['predictions'])

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
            logger.info("AZURE ENDPOINT PREDICTION METRICS")
            logger.info("=" * 50)
            logger.info(f"MSE: {mse:.4f}")
            logger.info(f"RMSE: {np.sqrt(mse):.4f}")
            logger.info(f"MAE: {mae:.4f}")
            logger.info(f"R² Score: {r2:.4f}")
            logger.info("=" * 50)

        # Save results
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(output_path, index=False)
            logger.info(f"Results saved to: {output_path}")

        return results_df

    def test_endpoint(self):
        """Test endpoint with sample data"""
        logger.info("Testing endpoint with sample data...")

        test_data = {
            "features": [
                [0.714, -1.515, -0.223, 0.617, -0.015, -0.366, 0.071, 1.095, -0.073, 0.626],
                [-0.234, 0.543, -0.234, -0.047, 0.028, 1.365, -0.457, 1.004, 0.767, 0.497]
            ]
        }

        result = self.predict(test_data)

        print("\n" + "=" * 60)
        print("ENDPOINT TEST RESULTS")
        print("=" * 60)
        print(f"Endpoint:     {self.scoring_uri}")
        print(f"Samples:      {result.get('num_samples')}")
        print(f"Model Type:   {result.get('model_type')}")
        print(f"Predictions:  {result.get('predictions')}")
        print("=" * 60)

        return result


def interactive_mode(client: AzureMLInference):
    """Interactive prediction mode"""
    logger.info("\n" + "=" * 60)
    logger.info("AZURE ML INTERACTIVE PREDICTION MODE")
    logger.info("=" * 60)
    logger.info("Enter 10 feature values (comma-separated)")
    logger.info("Enter 'q' to quit\n")

    while True:
        try:
            user_input = input("> ").strip()

            if user_input.lower() == 'q':
                logger.info("Exiting...")
                break

            # Parse input
            features = [float(x.strip()) for x in user_input.split(',')]

            if len(features) != 10:
                print(f"Error: Expected 10 features, got {len(features)}")
                continue

            # Make prediction
            prediction = client.predict_single(features)

            print(f"\n{'=' * 50}")
            print(f"Input: {features}")
            print(f"Prediction: {prediction:.4f}")
            print(f"{'=' * 50}\n")

        except ValueError as e:
            print(f"Error: Invalid input - {str(e)}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Make predictions using Azure ML endpoint',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test endpoint
  python inference_azure.py --test

  # Interactive mode
  python inference_azure.py --interactive

  # Predict from CSV
  python inference_azure.py --csv data/test_sample.csv

  # Predict from CSV and save
  python inference_azure.py --csv data/test.csv --output predictions.csv

  # Predict single instance
  python inference_azure.py --features 0.7,-1.5,-0.2,0.6,-0.01,-0.4,0.07,1.1,-0.07,0.6

  # Use specific endpoint
  python inference_azure.py --endpoint my-endpoint --test
        """
    )

    parser.add_argument(
        '--endpoint',
        type=str,
        help='Endpoint name (loads config from file)'
    )
    parser.add_argument(
        '--scoring-uri',
        type=str,
        help='Endpoint scoring URI'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='Endpoint API key'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test endpoint with sample data'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    parser.add_argument(
        '--csv',
        type=str,
        help='Path to CSV file for batch prediction'
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

    args = parser.parse_args()

    try:
        # Initialize client
        client = AzureMLInference(
            scoring_uri=args.scoring_uri,
            api_key=args.api_key,
            endpoint_name=args.endpoint
        )

        # Execute requested operation
        if args.test:
            client.test_endpoint()

        elif args.csv:
            results = client.predict_from_csv(args.csv, args.output)
            print("\n" + "=" * 60)
            print("PREDICTION RESULTS (first 10 rows)")
            print("=" * 60)
            print(results.head(10).to_string(index=False))
            print("=" * 60)
            print(f"\nTotal predictions: {len(results)}")
            if args.output:
                print(f"Full results saved to: {args.output}")

        elif args.features:
            features = [float(x.strip()) for x in args.features.split(',')]
            prediction = client.predict_single(features)
            print("\n" + "=" * 60)
            print("PREDICTION RESULT")
            print("=" * 60)
            print(f"Input features: {features}")
            print(f"Prediction: {prediction:.4f}")
            print("=" * 60)

        elif args.interactive:
            interactive_mode(client)

        else:
            # Default: test endpoint
            client.test_endpoint()

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
