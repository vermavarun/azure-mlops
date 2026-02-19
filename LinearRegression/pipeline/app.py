"""
Azure MLOps Pipeline for Linear Regression
Main pipeline orchestration script
"""

import os
import logging
from datetime import datetime
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import command

from config import get_config
from data_handler import load_data, prepare_data
from train import train_model
from evaluate import evaluate_model


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


def run_local_pipeline():
    """Run pipeline locally without Azure ML"""
    logger.info("Starting local pipeline execution...")
    config = get_config()
    
    try:
        # Step 1: Load and prepare data
        logger.info("Step 1: Loading and preparing data...")
        X_train, X_test, y_train, y_test = load_data(
            test_size=config['test_size'],
            random_state=config['random_state']
        )
        
        logger.info(f"Training set size: {X_train.shape[0]}")
        logger.info(f"Test set size: {X_test.shape[0]}")
        
        # Step 2: Train model
        logger.info("Step 2: Training linear regression model...")
        model = train_model(X_train, y_train)
        logger.info("Model training completed")
        
        # Step 3: Evaluate model
        logger.info("Step 3: Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test)
        
        logger.info("=" * 50)
        logger.info("MODEL EVALUATION RESULTS")
        logger.info("=" * 50)
        logger.info(f"MSE: {metrics['mse']:.4f}")
        logger.info(f"RMSE: {metrics['rmse']:.4f}")
        logger.info(f"MAE: {metrics['mae']:.4f}")
        logger.info(f"R² Score: {metrics['r2_score']:.4f}")
        logger.info("=" * 50)
        
        # Save model
        import joblib
        model_path = Path("models") / f"linear_regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        logger.info(f"Model saved to: {model_path}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        raise


def run_azure_ml_pipeline():
    """Run pipeline on Azure ML"""
    logger.info("Starting Azure ML pipeline execution...")
    config = get_config()
    
    ml_client = get_ml_client()
    
    # Define component commands
    prepare_data_cmd = command(
        code="./src",
        command="python data_handler.py",
        environment="AzureML-sklearn-1.0",
        compute=config['compute_cluster_name'],
    )
    
    train_model_cmd = command(
        code="./src",
        command="python train.py",
        environment="AzureML-sklearn-1.0",
        compute=config['compute_cluster_name'],
    )
    
    evaluate_model_cmd = command(
        code="./src",
        command="python evaluate.py",
        environment="AzureML-sklearn-1.0",
        compute=config['compute_cluster_name'],
    )
    
    # Define pipeline
    @pipeline(
        default_compute=config['compute_cluster_name'],
        description="Linear Regression MLOps Pipeline"
    )
    def linear_regression_pipeline():
        """Azure ML Pipeline Definition"""
        
        prepare_job = prepare_data_cmd()
        train_job = train_model_cmd()
        eval_job = evaluate_model_cmd()
        
        return {
            "predictions": eval_job.outputs.predictions
        }
    
    # Create pipeline
    pipeline_job = linear_regression_pipeline()
    pipeline_job.display_name = f"linear-regression-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Submit pipeline
    logger.info(f"Submitting pipeline: {pipeline_job.display_name}")
    returned_job = ml_client.jobs.create_or_update(pipeline_job)
    
    logger.info(f"Pipeline job created: {returned_job.id}")
    logger.info(f"Pipeline status: {returned_job.status}")
    
    return returned_job


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--azure":
        # Run on Azure ML
        job = run_azure_ml_pipeline()
        logger.info(f"Pipeline submitted successfully. Job ID: {job.id}")
    else:
        # Run locally
        metrics = run_local_pipeline()
        logger.info("Pipeline completed successfully!")
