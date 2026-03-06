"""
Complete Azure ML Inference Workflow Example
Demonstrates end-to-end process from local training to Azure deployment and inference
"""

import os
import sys
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_1_local_training():
    """
    Example 1: Train model locally
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Local Model Training")
    print("=" * 70)

    from app import run_local_pipeline

    logger.info("Training model locally...")
    metrics = run_local_pipeline()

    logger.info(f"Training complete! R² Score: {metrics['r2_score']:.4f}")
    return metrics


def example_2_deploy_to_azure():
    """
    Example 2: Deploy model to Azure ML endpoint
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Deploy to Azure ML")
    print("=" * 70)

    from deploy_azure import deploy_model

    logger.info("Deploying model to Azure ML...")
    logger.info("This will take approximately 10-15 minutes...")

    # This deploys the latest trained model
    deploy_model()

    logger.info("Deployment complete!")


def example_3_test_endpoint():
    """
    Example 3: Test deployed endpoint
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Test Azure Endpoint")
    print("=" * 70)

    from inference_azure import AzureMLInference

    # Initialize client (automatically loads from outputs/endpoint_config.json)
    client = AzureMLInference()

    # Test with sample data
    result = client.test_endpoint()

    logger.info("Endpoint test passed!")
    return result


def example_4_single_prediction():
    """
    Example 4: Make single prediction via Azure endpoint
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Single Prediction via Azure")
    print("=" * 70)

    from inference_azure import AzureMLInference

    client = AzureMLInference()

    # Sample features (10 features for linear regression)
    features = [0.714, -1.515, -0.223, 0.617, -0.015,
                -0.366, 0.071, 1.095, -0.073, 0.626]

    prediction = client.predict_single(features)

    print(f"\nInput features: {features}")
    print(f"Prediction: {prediction:.4f}")

    return prediction


def example_5_batch_prediction():
    """
    Example 5: Batch prediction from CSV via Azure endpoint
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Batch Prediction via Azure")
    print("=" * 70)

    from inference_azure import AzureMLInference

    client = AzureMLInference()

    # Predict from CSV
    csv_path = 'data/test_sample.csv'
    output_path = 'data/azure_predictions.csv'

    if not Path(csv_path).exists():
        logger.warning(f"Test file not found: {csv_path}")
        logger.info("Run 'make local' first to generate test data")
        return None

    results_df = client.predict_from_csv(csv_path, output_path)

    print(f"\nResults preview (first 5 rows):")
    print(results_df[['prediction', 'target', 'error']].head())
    print(f"\nTotal predictions: {len(results_df)}")
    print(f"Results saved to: {output_path}")

    return results_df


def example_6_compare_local_vs_azure():
    """
    Example 6: Compare local inference vs Azure endpoint
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Compare Local vs Azure Inference")
    print("=" * 70)

    import numpy as np
    from inference import ModelInference
    from inference_azure import AzureMLInference

    # Sample data
    test_features = [
        [0.714, -1.515, -0.223, 0.617, -0.015, -0.366, 0.071, 1.095, -0.073, 0.626],
        [-0.234, 0.543, -0.234, -0.047, 0.028, 1.365, -0.457, 1.004, 0.767, 0.497]
    ]

    # Local inference
    print("\n1. Local Inference:")
    local_client = ModelInference()
    local_predictions = local_client.predict(np.array(test_features))
    print(f"   Predictions: {local_predictions}")

    # Azure inference
    print("\n2. Azure Inference:")
    azure_client = AzureMLInference()
    azure_result = azure_client.predict(test_features)
    azure_predictions = np.array(azure_result['predictions'])
    print(f"   Predictions: {azure_predictions}")

    # Compare
    print("\n3. Comparison:")
    diff = np.abs(local_predictions - azure_predictions)
    print(f"   Max difference: {np.max(diff):.6f}")
    print(f"   Mean difference: {np.mean(diff):.6f}")

    if np.allclose(local_predictions, azure_predictions, rtol=1e-5):
        print("   ✓ Predictions match!")
    else:
        print("   ⚠ Predictions differ (may be due to different model versions)")

    return local_predictions, azure_predictions


def example_7_programmatic_workflow():
    """
    Example 7: Complete programmatic workflow
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Programmatic Workflow")
    print("=" * 70)

    # Step 1: Generate synthetic data
    print("\nStep 1: Generating data...")
    from data_handler import generate_synthetic_data
    df = generate_synthetic_data(n_samples=50, n_features=10)
    df.to_csv('data/workflow_test.csv', index=False)
    print(f"   Generated {len(df)} samples")

    # Step 2: Load endpoint config
    print("\nStep 2: Loading Azure endpoint...")
    config_file = Path('outputs/endpoint_config.json')
    if not config_file.exists():
        print("   ⚠ Endpoint not deployed. Run 'make deploy-azure' first")
        return

    with open(config_file) as f:
        endpoint_config = json.load(f)
    print(f"   Endpoint: {endpoint_config['endpoint_name']}")

    # Step 3: Make predictions
    print("\nStep 3: Making predictions...")
    from inference_azure import AzureMLInference
    client = AzureMLInference()

    results_df = client.predict_from_csv(
        'data/workflow_test.csv',
        'data/workflow_predictions.csv'
    )

    # Step 4: Analyze results
    print("\nStep 4: Analysis:")
    print(f"   Total predictions: {len(results_df)}")
    print(f"   Mean prediction: {results_df['prediction'].mean():.2f}")
    print(f"   Std prediction: {results_df['prediction'].std():.2f}")
    print(f"   Min prediction: {results_df['prediction'].min():.2f}")
    print(f"   Max prediction: {results_df['prediction'].max():.2f}")

    if 'target' in results_df.columns:
        from sklearn.metrics import r2_score
        r2 = r2_score(results_df['target'], results_df['prediction'])
        print(f"   R² Score: {r2:.4f}")


def main():
    """
    Run all examples
    """
    print("\n" + "=" * 70)
    print("AZURE ML INFERENCE - COMPLETE WORKFLOW EXAMPLES")
    print("=" * 70)
    print("\nThis script demonstrates the complete workflow:")
    print("1. Train model locally")
    print("2. Deploy to Azure ML")
    print("3. Test endpoint")
    print("4. Single prediction")
    print("5. Batch prediction")
    print("6. Compare local vs Azure")
    print("7. Programmatic workflow")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--example',
        type=int,
        choices=range(1, 8),
        help='Run specific example (1-7), or all if not specified'
    )
    parser.add_argument(
        '--skip-deploy',
        action='store_true',
        help='Skip deployment step (assumes endpoint already exists)'
    )

    args = parser.parse_args()

    try:
        if args.example:
            # Run specific example
            if args.example == 1:
                example_1_local_training()
            elif args.example == 2:
                if not args.skip_deploy:
                    example_2_deploy_to_azure()
                else:
                    print("Skipping deployment (--skip-deploy)")
            elif args.example == 3:
                example_3_test_endpoint()
            elif args.example == 4:
                example_4_single_prediction()
            elif args.example == 5:
                example_5_batch_prediction()
            elif args.example == 6:
                example_6_compare_local_vs_azure()
            elif args.example == 7:
                example_7_programmatic_workflow()
        else:
            # Run all examples (skip deployment if flag set)
            example_1_local_training()

            if not args.skip_deploy:
                example_2_deploy_to_azure()
            else:
                print("\nSkipping deployment (--skip-deploy flag set)")

            example_3_test_endpoint()
            example_4_single_prediction()
            example_5_batch_prediction()
            example_6_compare_local_vs_azure()
            example_7_programmatic_workflow()

        print("\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 70)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.info("\nMake sure to:")
        logger.info("  1. Train model: make local")
        logger.info("  2. Deploy endpoint: make deploy-azure")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
