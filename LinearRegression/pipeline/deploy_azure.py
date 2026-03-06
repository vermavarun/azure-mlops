"""
Deploy model to Azure ML as a managed online endpoint
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration
)
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceExistsError

from config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_ml_client():
    """Initialize Azure ML Client"""
    config = get_config()

    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=config['subscription_id'],
        resource_group_name=config['resource_group_name'],
        workspace_name=config['workspace_name']
    )

    logger.info(f"Connected to workspace: {config['workspace_name']}")
    return ml_client


def register_model(ml_client: MLClient, model_path: str = None) -> Model:
    """
    Register model in Azure ML workspace

    Args:
        ml_client: Azure ML client
        model_path: Path to model file. If None, uses latest model.

    Returns:
        Registered Model object
    """
    config = get_config()

    if model_path is None:
        # Find latest model
        models_dir = Path(config['models_path'])
        model_files = list(models_dir.glob('*.pkl')) + list(models_dir.glob('*.joblib'))

        if not model_files:
            raise FileNotFoundError(
                f"No model files found in {models_dir}\n"
                "Please train a model first using 'make local'"
            )

        model_path = max(model_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Using latest model: {model_path}")
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Create model name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    model_name = f"{config['model_name']}-{timestamp}"

    logger.info(f"Registering model: {model_name}")

    model = Model(
        path=str(model_path),
        name=model_name,
        description="Linear Regression model for MLOps pipeline",
        tags={
            "framework": "scikit-learn",
            "model_type": "LinearRegression",
            "created_at": timestamp
        }
    )

    registered_model = ml_client.models.create_or_update(model)
    logger.info(f"Model registered: {registered_model.name} (version {registered_model.version})")

    return registered_model


def create_endpoint(ml_client: MLClient, endpoint_name: str = None) -> ManagedOnlineEndpoint:
    """
    Create or get Azure ML managed online endpoint

    Args:
        ml_client: Azure ML client
        endpoint_name: Name of endpoint. If None, uses config.

    Returns:
        ManagedOnlineEndpoint object
    """
    config = get_config()

    if endpoint_name is None:
        endpoint_name = os.getenv('ENDPOINT_NAME', f"{config['model_name']}-endpoint")

    # Ensure endpoint name is valid (lowercase, alphanumeric, hyphens)
    endpoint_name = endpoint_name.lower().replace('_', '-')

    logger.info(f"Creating endpoint: {endpoint_name}")

    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description="Endpoint for Linear Regression model inference",
        auth_mode="key",
        tags={
            "project": "mlops-linear-regression",
            "created_at": datetime.now().isoformat()
        }
    )

    try:
        endpoint_result = ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        logger.info(f"Endpoint created: {endpoint_result.name}")
    except ResourceExistsError:
        logger.info(f"Endpoint already exists: {endpoint_name}")
        endpoint_result = ml_client.online_endpoints.get(endpoint_name)

    return endpoint_result


def create_deployment(
    ml_client: MLClient,
    endpoint_name: str,
    model: Model,
    deployment_name: str = None
) -> ManagedOnlineDeployment:
    """
    Create deployment for the endpoint

    Args:
        ml_client: Azure ML client
        endpoint_name: Name of the endpoint
        model: Registered model
        deployment_name: Name of deployment. If None, uses 'blue'.

    Returns:
        ManagedOnlineDeployment object
    """
    if deployment_name is None:
        deployment_name = "blue"

    logger.info(f"Creating deployment: {deployment_name}")

    # Create environment for deployment
    env = Environment(
        name="linear-regression-serving-env",
        description="Environment for serving Linear Regression model",
        conda_file="conda.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04"
    )

    # Create deployment
    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model,
        environment=env,
        code_configuration=CodeConfiguration(
            code=".",
            scoring_script="score.py"
        ),
        instance_type="Standard_DS2_v2",
        instance_count=1,
        request_settings={
            "request_timeout_ms": 30000,
            "max_concurrent_requests_per_instance": 1,
            "max_queue_wait_ms": 60000
        }
    )

    logger.info("Starting deployment (this may take 10-15 minutes)...")
    deployment_result = ml_client.online_deployments.begin_create_or_update(deployment).result()
    logger.info(f"Deployment created: {deployment_result.name}")

    # Set deployment to receive 100% of traffic
    logger.info("Routing 100% traffic to deployment...")
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    endpoint.traffic = {deployment_name: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    return deployment_result


def get_endpoint_details(ml_client: MLClient, endpoint_name: str):
    """
    Get and display endpoint details including scoring URI and keys

    Args:
        ml_client: Azure ML client
        endpoint_name: Name of endpoint
    """
    logger.info(f"Fetching endpoint details for: {endpoint_name}")

    endpoint = ml_client.online_endpoints.get(endpoint_name)
    keys = ml_client.online_endpoints.get_keys(endpoint_name)

    print("\n" + "=" * 70)
    print("AZURE ML ENDPOINT DETAILS")
    print("=" * 70)
    print(f"Endpoint Name:    {endpoint.name}")
    print(f"Scoring URI:      {endpoint.scoring_uri}")
    print(f"Swagger URI:      {endpoint.openapi_uri}")
    print(f"Auth Mode:        {endpoint.auth_mode}")
    print(f"Provisioning:     {endpoint.provisioning_state}")
    print("\nPrimary Key:")
    print(f"  {keys.primary_key}")
    print("\nSecondary Key:")
    print(f"  {keys.secondary_key}")
    print("=" * 70)

    # Save to file for easy access
    endpoint_info = {
        "endpoint_name": endpoint.name,
        "scoring_uri": endpoint.scoring_uri,
        "primary_key": keys.primary_key,
        "secondary_key": keys.secondary_key,
        "created_at": datetime.now().isoformat()
    }

    import json
    output_file = Path("outputs") / "endpoint_config.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(endpoint_info, f, indent=2)

    logger.info(f"Endpoint details saved to: {output_file}")

    return endpoint, keys


def deploy_model(
    model_path: str = None,
    endpoint_name: str = None,
    deployment_name: str = None
):
    """
    Full deployment workflow: register model, create endpoint, create deployment

    Args:
        model_path: Path to model file
        endpoint_name: Name of endpoint
        deployment_name: Name of deployment
    """
    try:
        # Initialize Azure ML client
        ml_client = get_ml_client()

        # Step 1: Register model
        logger.info("\n" + "=" * 70)
        logger.info("STEP 1: Registering Model")
        logger.info("=" * 70)
        model = register_model(ml_client, model_path)

        # Step 2: Create endpoint
        logger.info("\n" + "=" * 70)
        logger.info("STEP 2: Creating Endpoint")
        logger.info("=" * 70)
        endpoint = create_endpoint(ml_client, endpoint_name)

        # Step 3: Create deployment
        logger.info("\n" + "=" * 70)
        logger.info("STEP 3: Creating Deployment")
        logger.info("=" * 70)
        deployment = create_deployment(ml_client, endpoint.name, model, deployment_name)

        # Step 4: Get endpoint details
        logger.info("\n" + "=" * 70)
        logger.info("STEP 4: Endpoint Details")
        logger.info("=" * 70)
        get_endpoint_details(ml_client, endpoint.name)

        print("\n" + "=" * 70)
        print("DEPLOYMENT COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nYou can now test the endpoint using:")
        print(f"  python inference_azure.py --endpoint {endpoint.name}")
        print("\nOr using make:")
        print(f"  make infer-azure ENDPOINT={endpoint.name}")
        print("=" * 70)

    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}", exc_info=True)
        raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Deploy Linear Regression model to Azure ML',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--model',
        type=str,
        help='Path to model file (default: latest model)'
    )
    parser.add_argument(
        '--endpoint',
        type=str,
        help='Endpoint name (default: from config)'
    )
    parser.add_argument(
        '--deployment',
        type=str,
        default='blue',
        help='Deployment name (default: blue)'
    )

    args = parser.parse_args()

    deploy_model(
        model_path=args.model,
        endpoint_name=args.endpoint,
        deployment_name=args.deployment
    )


if __name__ == "__main__":
    main()
