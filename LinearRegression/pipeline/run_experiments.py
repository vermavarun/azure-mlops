#!/usr/bin/env python
"""
Experiment runner for systematic model training and evaluation.

This script enables running multiple experiments with different configurations,
hyperparameters, and data splits, with automatic result tracking and comparison.

Usage:
    python run_experiments.py --experiment baseline
    python run_experiments.py --experiment all
    python run_experiments.py --list
"""

import json
import logging
from datetime import datetime
from pathlib import Path
import argparse

from config import get_config
from data_handler import load_data
from train import train_model, save_model
from evaluate import evaluate_model, save_metrics
from experiment_configs import get_experiment_config, list_experiments

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Run and track experiments."""
    
    def __init__(self):
        """Initialize experiment runner."""
        self.config = get_config()
        self.results = {}
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = Path('experiments') / self.timestamp
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def run_experiment(self, experiment_name):
        """Run a single experiment.
        
        Args:
            experiment_name: Name of experiment from experiment_configs
            
        Returns:
            Dictionary with experiment results
        """
        logger.info(f"Starting experiment: {experiment_name}")
        
        try:
            # Load experiment config
            exp_config = get_experiment_config(experiment_name)
            logger.info(f"Config: {exp_config['name']}")
            
            # Load and prepare data
            logger.info("Loading data...")
            X_train, X_test, y_train, y_test = load_data(
                **exp_config['data_config']
            )
            logger.info(f"Data: {X_train.shape[0]} train, {X_test.shape[0]} test")
            
            # Train model
            model_type = exp_config['model_type']
            model_kwargs = exp_config['model_kwargs']
            logger.info(f"Training {model_type} model with {model_kwargs}")
            
            model, train_score = train_model(
                X_train, y_train,
                model_type=model_type,
                **model_kwargs
            )
            logger.info(f"Train R²: {train_score:.4f}")
            
            # Evaluate model
            logger.info("Evaluating model...")
            metrics = evaluate_model(model, X_test, y_test)
            logger.info(f"Test R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")
            
            # Save results
            result = {
                'experiment': experiment_name,
                'config': exp_config,
                'metrics': metrics,
                'train_score': train_score,
            }
            
            # Save model and metrics
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = self.results_dir / f"{model_type}_{timestamp}.pkl"
            metrics_path = self.results_dir / f"{model_type}_{timestamp}_metrics.json"
            
            save_model(model, str(model_path))
            save_metrics(metrics, str(metrics_path))
            
            result['model_path'] = str(model_path)
            result['metrics_path'] = str(metrics_path)
            
            logger.info(f"✓ Experiment {experiment_name} complete")
            return result
            
        except Exception as e:
            logger.error(f"✗ Experiment {experiment_name} failed: {e}", exc_info=True)
            return {'experiment': experiment_name, 'error': str(e)}
    
    def run_all_experiments(self):
        """Run all available experiments.
        
        Returns:
            Dictionary mapping experiment names to results
        """
        logger.info("Running all experiments...")
        experiments = list_experiments()
        
        for exp_name in experiments:
            self.results[exp_name] = self.run_experiment(exp_name)
        
        return self.results
    
    def compare_results(self):
        """Display comparison of experiment results.
        
        Returns:
            Formatted comparison table
        """
        logger.info("\n" + "="*80)
        logger.info("EXPERIMENT COMPARISON")
        logger.info("="*80)
        
        # Collect results
        comparison = []
        for exp_name, result in self.results.items():
            if 'error' in result:
                logger.warning(f"{exp_name}: ERROR - {result['error']}")
                continue
            
            metrics = result.get('metrics', {})
            comparison.append({
                'Experiment': exp_name,
                'Model': result['config']['model_type'],
                'R² Score': f"{metrics.get('r2', 0):.4f}",
                'RMSE': f"{metrics.get('rmse', 0):.4f}",
                'MAE': f"{metrics.get('mae', 0):.4f}",
                'MAPE': f"{metrics.get('mape', 0):.4f}",
            })
        
        # Display table
        if comparison:
            print("\n" + "="*100)
            headers = list(comparison[0].keys())
            print(" | ".join(f"{h:20}" for h in headers))
            print("-"*100)
            for row in comparison:
                print(" | ".join(f"{str(row[h]):20}" for h in headers))
            print("="*100 + "\n")
        
        return comparison
    
    def save_summary(self):
        """Save comprehensive summary of all experiments."""
        summary = {
            'timestamp': self.timestamp,
            'results_dir': str(self.results_dir),
            'experiments': self.results,
        }
        
        summary_path = self.results_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Summary saved to {summary_path}")
        return summary_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run MLOps experiments'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        help='Experiment name (or "all" for all experiments)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available experiments'
    )
    
    args = parser.parse_args()
    
    runner = ExperimentRunner()
    
    if args.list:
        print("Available experiments:")
        for exp in list_experiments():
            print(f"  - {exp}")
    elif args.experiment:
        if args.experiment.lower() == 'all':
            runner.run_all_experiments()
        else:
            runner.results[args.experiment] = runner.run_experiment(
                args.experiment
            )
        
        # Display comparison and save
        runner.compare_results()
        runner.save_summary()
    else:
        # Default: run comparison experiment
        runner.results['comparison'] = runner.run_experiment('comparison')
        runner.compare_results()
        runner.save_summary()


if __name__ == '__main__':
    main()
