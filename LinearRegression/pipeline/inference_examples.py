"""
Example script demonstrating programmatic use of the inference module
"""

import numpy as np
from inference import ModelInference


def example_single_prediction():
    """Example: Single prediction"""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Single Prediction")
    print("=" * 60)

    # Initialize inference handler (loads latest model)
    inference = ModelInference()

    # Sample feature values (10 features for this model)
    features = [0.714, -1.515, -0.223, 0.617, -0.015,
                -0.366, 0.071, 1.095, -0.073, 0.626]

    # Make prediction
    prediction = inference.predict_single(features)

    print(f"Input features: {features}")
    print(f"Prediction: {prediction:.4f}")


def example_batch_prediction():
    """Example: Batch prediction with NumPy array"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Batch Prediction (NumPy)")
    print("=" * 60)

    # Initialize inference handler
    inference = ModelInference()

    # Create batch of samples
    X_batch = np.array([
        [0.714, -1.515, -0.223, 0.617, -0.015, -0.366, 0.071, 1.095, -0.073, 0.626],
        [-0.234, 0.543, -0.234, -0.047, 0.028, 1.365, -0.457, 1.004, 0.767, 0.497],
        [0.404, -0.074, -0.161, -0.213, -0.426, -0.449, 0.027, 0.107, 0.175, -1.415]
    ])

    # Make predictions
    predictions = inference.predict(X_batch)

    print(f"Input shape: {X_batch.shape}")
    print(f"Predictions: {predictions}")
    print(f"Number of predictions: {len(predictions)}")


def example_csv_prediction():
    """Example: Prediction from CSV file"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: CSV File Prediction")
    print("=" * 60)

    # Initialize inference handler
    inference = ModelInference()

    # Predict from CSV (using sample data)
    csv_path = 'data/test_sample.csv'
    output_path = 'data/predictions_output.csv'

    print(f"Reading from: {csv_path}")
    results_df = inference.predict_from_csv(csv_path, output_path)

    print(f"\nPrediction results (first 5):")
    print(results_df[['prediction', 'target', 'error', 'abs_error']].head())
    print(f"\nResults saved to: {output_path}")


def example_json_prediction():
    """Example: Prediction from JSON data"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: JSON Prediction")
    print("=" * 60)

    # Initialize inference handler
    inference = ModelInference()

    # Method 1: From JSON file
    json_path = 'data/test_sample.json'
    print(f"Reading from: {json_path}")
    results = inference.predict_from_json(json_path)
    print(f"Predictions: {results['predictions']}")
    print(f"Number of samples: {results['num_samples']}")

    # Method 2: From dictionary
    print("\nUsing dictionary input:")
    data = {
        "features": [
            [0.714, -1.515, -0.223, 0.617, -0.015, -0.366, 0.071, 1.095, -0.073, 0.626]
        ]
    }
    results = inference.predict_from_json(data)
    print(f"Predictions: {results['predictions']}")


def example_specific_model():
    """Example: Using a specific model file"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Using Specific Model")
    print("=" * 60)

    # Find model files
    from pathlib import Path
    models_dir = Path('models')
    model_files = list(models_dir.glob('*.pkl'))

    if model_files:
        # Use first model found
        model_path = str(model_files[0])
        print(f"Loading model: {model_path}")

        # Initialize with specific model
        inference = ModelInference(model_path=model_path)

        # Make prediction
        features = [0.714, -1.515, -0.223, 0.617, -0.015,
                   -0.366, 0.071, 1.095, -0.073, 0.626]
        prediction = inference.predict_single(features)

        print(f"Prediction: {prediction:.4f}")
    else:
        print("No model files found. Train a model first with 'make local'")


def example_error_handling():
    """Example: Error handling"""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Error Handling")
    print("=" * 60)

    try:
        inference = ModelInference()

        # Try prediction with wrong number of features
        print("Attempting prediction with wrong number of features...")
        features = [1.0, 2.0, 3.0]  # Only 3 features instead of 10
        prediction = inference.predict_single(features)

    except ValueError as e:
        print(f"ValueError caught: {e}")
    except Exception as e:
        print(f"Error caught: {type(e).__name__}: {e}")


def example_with_custom_processing():
    """Example: Custom pre/post processing"""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Custom Processing Pipeline")
    print("=" * 60)

    inference = ModelInference()

    # Simulate custom preprocessing
    raw_data = np.random.randn(5, 10)

    # Custom preprocessing (e.g., scaling, normalization)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_processed = scaler.fit_transform(raw_data)

    print(f"Raw data shape: {raw_data.shape}")
    print(f"Processed data shape: {X_processed.shape}")

    # Make predictions
    predictions = inference.predict(X_processed)

    # Custom postprocessing (e.g., inverse transform, rounding)
    predictions_rounded = np.round(predictions, 2)

    print(f"Raw predictions: {predictions}")
    print(f"Rounded predictions: {predictions_rounded}")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("INFERENCE MODULE - USAGE EXAMPLES")
    print("=" * 60)
    print("\nThese examples demonstrate various ways to use the inference module")
    print("Make sure you have a trained model first (run 'make local')")

    try:
        # Run examples
        example_single_prediction()
        example_batch_prediction()

        # Check if sample data exists
        from pathlib import Path
        if Path('data/test_sample.csv').exists():
            example_csv_prediction()
        else:
            print("\nSkipping CSV example - sample file not found")

        if Path('data/test_sample.json').exists():
            example_json_prediction()
        else:
            print("\nSkipping JSON example - sample file not found")

        example_specific_model()
        example_error_handling()
        example_with_custom_processing()

        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease train a model first:")
        print("  make local")
        print("\nOr run:")
        print("  python app.py")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
