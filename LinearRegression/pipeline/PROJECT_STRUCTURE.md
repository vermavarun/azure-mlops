# Project Structure & File Guide

Complete reference for all files in the MLOps Linear Regression Pipeline.

## Directory Structure

```
LinearRegression/
├── Infra/                              # Terraform Infrastructure (DEPLOYED)
│   ├── providers.tf                    # Azure provider configuration
│   ├── main.tf                         # Main infrastructure resources
│   ├── variables.tf                    # Variable definitions
│   ├── outputs.tf                      # Resource outputs
│   └── terraform.tfvars.example        # Example terraform variables
│
└── pipeline/                           # MLOps Pipeline Code
    ├── Core Modules
    │   ├── app.py                      # ⭐ Main entry point (orchestration)
    │   ├── config.py                   # Configuration management
    │   ├── data_handler.py             # Data loading/preprocessing
    │   ├── train.py                    # Model training
    │   └── evaluate.py                 # Model evaluation/metrics
    │
    ├── Examples & Experiments
    │   ├── demo.py                     # 5 complete workflow examples
    │   ├── run_experiments.py           # Systematic experiment runner
    │   └── experiment_configs.py        # Pre-configured experiments
    │
    ├── Documentation
    │   ├── README.md                   # Complete guide (550+ lines)
    │   ├── QUICKSTART.md               # 5-minute quick start guide
    │   ├── CONTRIBUTING.md             # Contribution guidelines
    │   └── PROJECT_STRUCTURE.md        # This file
    │
    ├── Configuration & Setup
    │   ├── requirements.txt             # Core dependencies (9 packages)
    │   ├── requirements-dev.txt         # Dev tools (black, pytest, etc.)
    │   ├── .env.example                # Environment variable template
    │   └── .gitignore                  # Git exclusion patterns
    │
    ├── Automation Scripts
    │   ├── pipeline.ps1                # PowerShell helper (Windows)
    │   ├── pipeline.sh                 # Bash helper (Linux/macOS)
    │   └── Makefile                    # Make commands
    │
    ├── Docker Configuration
    │   ├── Dockerfile                  # Container definition
    │   └── docker-compose.yml          # Multi-container orchestration
    │
    ├── CI/CD Pipeline
    │   └── .github/workflows/
    │       ├── ci-cd.yml               # Automated testing & deployment
    │       └── deploy.yml              # Manual Azure ML deployment
    │
    ├── VS Code Configuration
    │   └── .vscode/
    │       ├── launch.json             # Python debug configurations
    │       ├── settings.json           # VS Code settings
    │       └── extensions.json         # Recommended extensions
    │
    ├── Output Directories (git-ignored)
    │   ├── data/                       # Generated training data
    │   ├── models/                     # Trained model files (.pkl)
    │   ├── metrics/                    # Evaluation metrics & plots
    │   ├── experiments/                # Experiment run results
    │   └── logs/                       # Application logs
    │
    └── Root Files
        └── LICENSE                     # MIT License
```

## Core Module Reference

### 1. **app.py** - Pipeline Orchestrator
**Purpose**: Main entry point for pipeline execution  
**Lines**: 151  
**Key Functions**:
- `get_ml_client()` - Azure ML authentication
- `run_local_pipeline()` - Local execution
- `run_azure_ml_pipeline()` - Cloud submission
- `main()` - CLI with --azure flag

**Usage**:
```bash
python app.py              # Run locally
python app.py --azure      # Run on Azure ML
```

### 2. **config.py** - Configuration Manager
**Purpose**: Environment-based configuration  
**Lines**: 71  
**Key Functions**:
- `get_config()` - Load 27 configuration parameters
- `save_config()` - Export to JSON
- `load_config_from_file()` - Import from JSON

**Configuration Keys**: subscription_id, workspace_name, compute_target, model_type, n_samples, test_size, etc.

**Usage**:
```python
from config import get_config
cfg = get_config()
print(cfg['workspace_name'])
```

### 3. **data_handler.py** - Data Pipeline
**Purpose**: Data loading, preprocessing, validation  
**Lines**: 218  
**Key Functions**:
- `generate_synthetic_data()` - Create synthetic dataset
- `load_data()` - Load CSV or synthetic data
- `prepare_data()` - Data cleaning & validation
- `save_data()` - Export splits as NumPy arrays

**Usage**:
```python
from data_handler import load_data
X_train, X_test, y_train, y_test = load_data()
```

### 4. **train.py** - Model Training
**Purpose**: Support 4 regression model types  
**Lines**: 196  
**Supported Models**:
- Linear Regression
- Polynomial Regression
- Ridge Regression (L2)
- Lasso Regression (L1)

**Key Functions**:
- `train_model()` - Factory function
- `train_linear_regression()` - Linear model
- `train_polynomial_regression()` - Poly features
- `train_ridge_regression()` - L2 regularization
- `train_lasso_regression()` - L1 regularization
- `save_model()` / `load_model()` - Serialization

**Usage**:
```python
from train import train_model, save_model
model, score = train_model(X_train, y_train, model_type='ridge', alpha=1.0)
save_model(model, 'my_model.pkl')
```

### 5. **evaluate.py** - Model Evaluation
**Purpose**: Metrics computation, cross-validation, visualization  
**Lines**: 272  
**Metrics**: MSE, RMSE, MAE, MAPE, R², residual stats  
**Key Functions**:
- `evaluate_model()` - Compute 8+ metrics
- `evaluate_cross_validation()` - K-fold CV
- `plot_predictions()` - Scatter + residual plots
- `plot_residual_distribution()` - Histogram + Q-Q
- `save_metrics()` / `load_metrics()` - JSON persistence

**Usage**:
```python
from evaluate import evaluate_model, plot_predictions
metrics = evaluate_model(model, X_test, y_test)
plot_predictions(model, X_test, y_test, output_path='plot.png')
```

## Example Scripts

### **demo.py** - 5 Complete Workflows
**Purpose**: Demonstrate all capabilities  
**Lines**: 280  
**Demonstrations**:
1. Basic workflow (train→eval→save)
2. Model comparison (3 model types)
3. Cross-validation (5-fold)
4. Data persistence (save/load)
5. Polynomial features (degree 1-3)

**Usage**:
```bash
python demo.py
```

### **run_experiments.py** - Experiment Runner
**Purpose**: Systematic model training with multiple configurations  
**Lines**: 260  
**Features**:
- Run individual or all experiments
- Automatic result tracking
- Comparison tables
- Summary export

**Usage**:
```bash
python run_experiments.py --list                 # Show available
python run_experiments.py --experiment baseline  # Run one
python run_experiments.py --experiment all       # Run all
```

### **experiment_configs.py** - Configuration Templates
**Purpose**: Pre-configured experiment scenarios  
**Lines**: 180  
**Experiments**:
- BASELINE: Simple linear regression
- POLYNOMIAL: Polynomial features with degree 2
- RIDGE: L2 regularization
- LASSO: L1 regularization with feature selection
- LARGE_SCALE: 1000 samples, 50 features
- COMPARISON: Side-by-side model comparison

**Usage**:
```python
from experiment_configs import get_experiment_config
cfg = get_experiment_config('ridge')
```

## Configuration Files

### **requirements.txt** - Core Dependencies
```
azure-ai-ml==1.10.0
azure-identity==1.13.0
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
scipy==1.11.1
joblib==1.3.1
python-dotenv==1.0.0
```

### **requirements-dev.txt** - Development Tools
Extends requirements.txt with:
- black (code formatter)
- pylint (linter)
- pytest (testing)
- mypy (type checker)
- jupyter (notebooks)

### **.env.example** - Environment Template
```
AZURE_SUBSCRIPTION_ID=1f8817ba-4f67-4ce6-9b70-e4b60fd2c3db
AZURE_WORKSPACE_NAME=mlops-lr-dev-amlws-45636e32
AZURE_RESOURCE_GROUP=mlops-lr-dev-rg-da8715cb
...
```

### **.gitignore** - Git Patterns
Ignores: `__pycache__/`, `*.pyc`, `data/`, `models/`, `metrics/`, `.env`, `venv/`

## Automation Scripts

### **pipeline.ps1** - PowerShell Helper (Windows)
**Commands**:
- `setup` - Create directories
- `install` - Install dependencies
- `local` - Run locally
- `azure` - Run on Azure ML
- `demo` - Run examples

**Usage**:
```powershell
.\pipeline.ps1 -command local
.\pipeline.ps1 -command demo
```

### **pipeline.sh** - Bash Helper (Linux/macOS)
**Same commands as pipeline.ps1**

**Usage**:
```bash
chmod +x pipeline.sh
./pipeline.sh local
./pipeline.sh demo
```

### **Makefile** - Make Commands
**Targets**:
```bash
make help          # Show all commands
make setup         # Initialize
make install       # Install deps
make local         # Run locally
make azure         # Run on cloud
make demo          # Run examples
make clean         # Remove outputs
make format        # Format code
make lint          # Check code
make test          # Run tests
```

## Docker Configuration

### **Dockerfile**
- Base: python:3.10-slim
- Installs dependencies
- Exposes ports for Jupyter

**Build**:
```bash
docker build -t mlops-lr-pipeline .
```

**Run**:
```bash
docker run --env-file .env mlops-lr-pipeline
docker run --env-file .env mlops-lr-pipeline python app.py --azure
```

### **docker-compose.yml**
- Pipeline service
- Optional Jupyter Lab service (port 8888)

**Usage**:
```bash
docker-compose up                    # Run all  
docker-compose up jupyter            # Jupyter only
docker-compose exec mlops-pipeline bash  # Shell
```

## CI/CD Pipeline

### **.github/workflows/ci-cd.yml**
Runs on: push to main/develop, pull requests

**Jobs**:
1. **test** - Unit tests, linting, import checks
2. **deploy** - Azure ML submission (main only)

### **.github/workflows/deploy.yml**
Manual workflow for deployment to dev/prod

**Trigger**: Actions → Deploy to Azure ML → Run workflow

## VS Code Configuration

### **.vscode/launch.json**
Debug configurations:
- Local Pipeline
- Azure Pipeline
- Demo Script
- Current File

### **.vscode/settings.json**
- Black formatter (100 char lines)
- Pylint linter
- Type checking
- File exclusions

### **.vscode/extensions.json**
Recommended extensions:
- Python, Pylance, Debugpy
- Ruff, Docker
- Azure Tools, GitLens

## Documentation Files

### **README.md** (550+ lines)
Complete guide covering:
- Installation & setup
- Architecture overview
- Module descriptions
- Model implementations
- Usage examples
- Azure ML integration
- Monitoring & logging
- Troubleshooting

### **QUICKSTART.md** (this file)
5-minute quick start with:
- Installation steps
- Configuration
- Running pipeline
- Troubleshooting

### **CONTRIBUTING.md**
Contributor guidelines:
- Development workflow
- Code style (Black/Pylint)
- Testing procedures
- PR process
- Adding new models

## Output Directories

All created during execution (git-ignored):

```
data/
├── synthetic_*.csv         # Generated training data
└── splits/                 # Train/test splits (.npy)

models/
├── linear_regression_*.pkl # Trained models
├── ridge_regression_*.pkl
└── ...

metrics/
├── linear_regression_*.json    # Evaluation metrics
├── linear_regression_*.png     # Plots (predictions, residuals)
└── ...

experiments/
└── YYYYMMDD_HHMMSS/        # Experiment run results
    ├── model_*.pkl
    ├── *_metrics.json
    └── summary.json

logs/
└── *.log                    # Application logs
```

## Development Workflow

### For Users:
1. Clone/download repository
2. `pip install -r requirements.txt`
3. `cp .env.example .env` (configure)
4. `python app.py` (run locally)
5. `python demo.py` (explore examples)

### For Developers:
1. Fork repository
2. `pip install -r requirements-dev.txt`
3. Create feature branch
4. `make format && make lint` (code quality)
5. `python -m pytest` (run tests)
6. Submit pull request

### For Data Scientists:
1. Use `demo.py` to understand pipeline
2. Modify `experiment_configs.py` for your settings
3. Run `python run_experiments.py` for systematic testing
4. Review metrics in `metrics/` and `experiments/`

## File Statistics

| Category | Count | Total Lines |
|----------|-------|-------------|
| Core Modules | 5 | 953 |
| Examples/Scripts | 3 | 540 |
| Documentation | 3 | 800+ |
| Configuration | 7 | 180 |
| Automation | 3 | 110 |
| Docker | 2 | 50 |
| CI/CD | 2 | 150 |
| VS Code | 3 | 80 |
| **Total** | **31** | **2,850+** |

## Quick Reference by Use Case

### "I want to run a quick test"
→ `python demo.py`

### "I want to use my data"
→ Edit `config.py`, set `use_synthetic=False`, provide CSV

### "I want to compare models"
→ `python run_experiments.py --experiment comparison`

### "I want to deploy to Azure"
→ `python app.py --azure`

### "I want to modify the code"
→ Read CONTRIBUTING.md

### "I want to containerize"
→ `docker build -t mlops-lr . && docker run ...`

### "I want CI/CD"
→ Push to GitHub, configure GitHub Secrets

---

**Last Updated**: 2024  
**License**: MIT  
**Authors**: MLOps Community
