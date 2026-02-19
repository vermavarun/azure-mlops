# Linear Regression MLOps Pipeline

A complete MLOps pipeline for training, evaluating, and deploying linear regression models on Azure Machine Learning.

## Directory Structure

```
pipeline/
├── app.py              # Main pipeline orchestration script
├── config.py           # Configuration management
├── data_handler.py     # Data loading and preprocessing
├── train.py            # Model training functions
├── evaluate.py         # Model evaluation and metrics
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── data/               # Input data directory
├── models/             # Trained models output
├── metrics/            # Evaluation metrics and visualizations
└── .env                # Environment variables (not committed)
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the pipeline directory:

```bash
AZURE_SUBSCRIPTION_ID=your_subscription_id
AZURE_RESOURCE_GROUP=mlops-lr-dev-rg-da8715cb
AZURE_WORKSPACE_NAME=mlops-lr-dev-amlws-45636e32
COMPUTE_CLUSTER=mlops-lr-dev-cc
COMPUTE_INSTANCE=mlops-lr-dev-ci
TEST_SIZE=0.2
RANDOM_STATE=42
```

Or set these environment variables directly in your shell/terminal.

## Usage

### Local Pipeline Execution

Run the pipeline locally without Azure ML:

```bash
python app.py
```

This will:
1. Load and prepare synthetic data (or from file if configured)
2. Train a linear regression model
3. Evaluate model performance
4. Save metrics and model artifacts

### Azure ML Pipeline Execution

Run the pipeline on Azure ML Compute Cluster:

```bash
python app.py --azure
```

This will:
1. Submit the pipeline to Azure ML
2. Execute training on the compute cluster
3. Track experiments and metrics in Azure ML

## Pipeline Components

### `app.py` - Main Orchestration
The main pipeline script that:
- Initializes Azure ML Client
- Orchestrates data preparation, training, and evaluation
- Supports both local and Azure ML execution
- Logs all operations

**Usage:**
```bash
python app.py          # Run locally
python app.py --azure  # Run on Azure ML
```

### `config.py` - Configuration Management
Manages all pipeline configuration from environment variables.

**Key Configurations:**
- Azure credentials and workspace settings
- Data parameters (test size, random state)
- Model parameters (fit intercept, normalization)
- Output paths for models and metrics

**Usage:**
```python
from config import get_config

config = get_config()
workspace_name = config['workspace_name']
test_size = config['test_size']
```

### `data_handler.py` - Data Preparation
Handles data loading, preprocessing, and validation.

**Functions:**
- `generate_synthetic_data()` - Create synthetic regression dataset
- `load_data()` - Load and split data into train/test sets
- `prepare_data()` - Handle outliers and validate data
- `save_data()` - Export data to files

**Example:**
```python
from data_handler import load_data

X_train, X_test, y_train, y_test = load_data(
    test_size=0.2,
    random_state=42,
    use_synthetic=True,
    scaling=True
)
```

### `train.py` - Model Training
Implements multiple regression models.

**Supported Models:**
- `train_linear_regression()` - Standard OLS regression
- `train_polynomial_regression()` - Polynomial regression
- `train_ridge_regression()` - Ridge regression (L2 regularization)
- `train_lasso_regression()` - Lasso regression (L1 regularization)

**Example:**
```python
from train import train_model

model = train_model(
    X_train, y_train,
    model_type='linear',
    fit_intercept=True
)
```

### `evaluate.py` - Model Evaluation
Evaluates model performance and generates visualizations.

**Functions:**
- `evaluate_model()` - Compute performance metrics
- `evaluate_cross_validation()` - K-fold cross-validation
- `plot_predictions()` - Visualize predictions vs actual
- `plot_residual_distribution()` - Analyze residuals
- `save_metrics()` - Export metrics to JSON

**Metrics Computed:**
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)
- R² Score
- Residual statistics

**Example:**
```python
from evaluate import evaluate_model, plot_predictions

metrics = evaluate_model(model, X_test, y_test)
plot_predictions(y_test, model.predict(X_test))
```

## Model Types

### Linear Regression
Simple least squares regression.
```python
model = train_linear_regression(X_train, y_train)
```

### Polynomial Regression
Fits polynomial features up to specified degree.
```python
poly_features, model = train_polynomial_regression(
    X_train, y_train, degree=2
)
```

### Ridge Regression
L2 regularization to prevent overfitting.
```python
model = train_ridge_regression(X_train, y_train, alpha=1.0)
```

### Lasso Regression
L1 regularization for feature selection.
```python
model = train_lasso_regression(X_train, y_train, alpha=0.1)
```

## Output Files

### Models
Trained models are saved in `models/` directory:
```
models/
├── linear_regression_20240219_143022.pkl
├── polynomial_regression_20240219_143535.pkl
└── ridge_regression_20240219_144012.pkl
```

### Metrics
Evaluation metrics saved in `metrics/` directory:
```
metrics/
├── metrics.json          # Numeric metrics
├── predictions.png       # Actual vs predicted plot
└── residuals.png         # Residual distribution plot
```

### Data
Processed data splits saved in `data/` directory:
```
data/
├── X_train.npy
├── X_test.npy
├── y_train.npy
└── y_test.npy
```

## Advanced Usage

### Custom Dataset
To use your own CSV data:

```python
from data_handler import load_data

X_train, X_test, y_train, y_test = load_data(
    use_synthetic=False,
    data_path='path/to/your/data.csv',
    test_size=0.2,
    scaling=True
)
```

Your CSV should have a `target` column for the labels.

### Cross-Validation
Evaluate model with cross-validation:

```python
from evaluate import evaluate_cross_validation

cv_results = evaluate_cross_validation(
    model, X, y, cv=5
)
print(f"R² Score: {cv_results['r2_mean']:.4f}")
```

### Custom Configuration
Create a custom config file:

```python
from config import save_config, load_config_from_file

# Save configuration
custom_config = {
    'test_size': 0.15,
    'random_state': 123,
    ...
}
save_config(custom_config, 'custom_config.json')

# Load configuration
config = load_config_from_file('custom_config.json')
```

## Azure ML Integration

### Register Model
```python
from azure.ai.ml.entities import Model

ml_client = get_ml_client()
model = Model(
    path="models/linear_regression.pkl",
    name="linear-regression-lr",
    version="1"
)
ml_client.models.create_or_update(model)
```

### Create Inference Endpoint
```python
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment

endpoint = ManagedOnlineEndpoint(name="linear-regression-endpoint")
ml_client.online_endpoints.begin_create_or_update(endpoint).wait()

deployment = ManagedOnlineDeployment(
    name="linear-regression-deploy",
    endpoint_name="linear-regression-endpoint",
    model=model,
    instance_type="Standard_F2s_v2",
    instance_count=1
)
ml_client.online_deployments.begin_create_or_update(deployment).wait()
```

## Monitoring & Logging

All operations are logged to console with:
- Timestamp
- Logger name
- Log level (INFO, WARNING, ERROR)
- Message

Example log output:
```
2024-02-19 14:30:22,123 - __main__ - INFO - Connected to workspace: mlops-lr-dev-amlws
2024-02-19 14:30:23,456 - data_handler - INFO - Loading data from: ./data/train.csv
2024-02-19 14:30:24,789 - train - INFO - Training Linear Regression model...
```

## Troubleshooting

### Authentication Issues
```bash
az login
az account set --subscription <subscription-id>
```

### Import Errors
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Data Loading Errors
- Check data file exists and is accessible
- Verify target column name matches configuration
- Ensure data has no NaN or infinity values

### Memory Errors
For large datasets, consider:
- Reducing batch size
- Using polynomial features with lower degree
- Downsampling data

## Performance Tips

1. **Data Scaling**: Always scale features for better convergence
2. **Feature Selection**: Remove highly correlated features
3. **Cross-Validation**: Use CV to validate model stability
4. **Regularization**: Use Ridge/Lasso to prevent overfitting
5. **Parallel Processing**: Azure ML automatically uses distributed computing

## Contributing

To extend the pipeline:

1. Add new model types in `train.py`
2. Add new metrics in `evaluate.py`
3. Extend configuration in `config.py`
4. Update requirements.txt with new dependencies

## License

This project is part of the Azure MLOps Linear Regression suite.

## Support

For issues and questions:
1. Check the Azure ML documentation
2. Review Azure ML Studio logs
3. Check pipeline logs in `./logs/` directory

## Further Reading

- [Azure ML Documentation](https://docs.microsoft.com/azure/machine-learning)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [MLOps Best Practices](https://azure.microsoft.com/en-us/resources/cloud-computing-dictionary/what-is-mlops/)
