# Azure MLOps - Linear Regression Pipeline

Complete MLOps pipeline for training and deploying linear regression models on Azure Machine Learning.

## Quick Links

📚 **Documentation:**
- [Quick Start Guide](LinearRegression/pipeline/QUICKSTART.md) - Get running in 5 minutes
- [Full README](LinearRegression/pipeline/README.md) - Complete guide with detailed setup
- [Project Structure](LinearRegression/pipeline/PROJECT_STRUCTURE.md) - File organization reference
- [Contributing Guide](LinearRegression/pipeline/CONTRIBUTING.md) - Development guidelines
- [Changelog](LinearRegression/pipeline/CHANGELOG.md) - Version history

## Project Structure

```
azure-mlops/
├── LinearRegression/
│   ├── Infra/          # Terraform infrastructure (Azure ML setup)
│   └── pipeline/       # MLOps pipeline code
└── .github/workflows/  # CI/CD GitHub Actions
```

## Core Pipeline

- **Training**: Scikit-learn linear regression models
- **Evaluation**: MSE, RMSE, MAE, R² metrics
- **Deployment**: Azure ML service integration
- **CI/CD**: Automated testing with GitHub Actions

## Getting Started

1. Read [Quick Start Guide](LinearRegression/pipeline/QUICKSTART.md)
2. Install dependencies and configure Azure credentials
3. Run locally: `python LinearRegression/pipeline/app.py`
4. Run on Azure ML: `python LinearRegression/pipeline/app.py --azure`

For detailed information, see the [Full README](LinearRegression/pipeline/README.md).
