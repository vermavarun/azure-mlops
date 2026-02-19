# Quick Start Guide

Get your MLOps pipeline running in 5 minutes!

## Prerequisites

- Python 3.8+ installed
- Azure subscription with credentials configured
- Git (optional)

## Step 1: Install Dependencies (2 minutes)

**Windows (PowerShell):**
```powershell
pip install -r requirements.txt
```

**Linux/macOS (Bash):**
```bash
pip install -r requirements.txt
```

Alternative with Makefile:
```bash
make install
```

## Step 2: Configure Azure (1 minute)

**Option A: Use defaults (local testing)**
```bash
cp .env.example .env
# Edit .env with your Azure credentials (optional)
```

**Option B: Only using Azure ML cloud**
```bash
az login
# Follow the prompts
cp .env.example .env
# Update AZURE_SUBSCRIPTION_ID, AZURE_WORKSPACE_NAME, etc.
```

## Step 3: Run Pipeline (2 minutes)

**Option A: Run locally with demo data**
```bash
python app.py
```

**Option B: Use automation helper (Windows)**
```powershell
.\pipeline.ps1 -command local
```

**Option C: Use automation helper (Linux/macOS)**
```bash
./pipeline.sh local
```

**Option D: Run comprehensive demos**
```bash
python demo.py
```

**Option E: Submit to Azure ML**
```bash
python app.py --azure
```

## Expected Output

### Local Run Output:
```
=== MLOps Linear Regression Pipeline ===
Loading configuration from environment variables...
✓ Configuration loaded successfully
◈ Data loading...
✓ Data loaded (80 training, 20 test samples)
◈ Model training...
✓ Model trained (Linear Regression) - Train R²: 0.95, Test R²: 0.92
◈ Model evaluation...
✓ Evaluation complete
✓ Model saved to models/linear_regression_20240101_120000.pkl
✓ Metrics saved to metrics/linear_regression_20240101_120000.json
✓ Plots saved to metrics/linear_regression_20240101_120000.png
```

### Demo Output:
```
========================================
MLOps Pipeline - Demonstration Script
========================================

Demo 1: Basic Workflow
[✓] Successfully trained and evaluated Linear Regression

Demo 2: Model Comparison
[✓] Comparison complete
├─ Linear Regression: R² = 0.945, RMSE = 2.15
├─ Ridge Regression: R² = 0.943, RMSE = 2.18
└─ Lasso Regression: R² = 0.932, RMSE = 2.45

... (more demos)
```

## File Structure

```
LinearRegression/pipeline/
├── app.py                     # Main entry point
├── config.py                  # Configuration management
├── data_handler.py            # Data operations
├── train.py                   # Model training
├── evaluate.py                # Metrics & visualization
├── demo.py                    # Example workflows
├── requirements.txt           # Core dependencies
├── requirements-dev.txt       # Dev tools
├── Makefile                   # Make commands
├── pipeline.ps1               # PowerShell helper
├── pipeline.sh                # Bash helper
├── README.md                  # Full documentation
├── .env.example               # Config template
├── .gitignore                 # Git patterns
├── data/                      # Input data (git-ignored)
├── models/                    # Trained models (git-ignored)
└── metrics/                   # Evaluation outputs (git-ignored)
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'azure'"
→ Run `make install` or `pip install -r requirements.txt`

### "Azure CLI not found" / Authentication errors
```bash
# Install Azure CLI
pip install azure-cli
# Or download from: https://aka.ms/AzureCLIDocs

# Authenticate
az login
az account set --subscription YOUR_SUBSCRIPTION_ID
```

### ".env file not found"
```bash
cp .env.example .env
# Then edit .env with your Azure credentials
```

### "AZURE_SUBSCRIPTION_ID not set" in environment
```bash
# Check your .env file exists and is in the pipeline directory
cat .env | grep AZURE_SUBSCRIPTION_ID
# Should output: AZURE_SUBSCRIPTION_ID=your_subscription_id
```

### Permission denied on .sh script (Linux/macOS)
```bash
chmod +x pipeline.sh
./pipeline.sh local
```

## Quick Command Reference

| Command | What it does |
|---------|-------------|
| `python app.py` | Run full pipeline locally |
| `python app.py --azure` | Submit job to Azure ML |
| `python demo.py` | Run 5 example scenarios |
| `python -c "from config import get_config; print(get_config())"` | Test configuration |
| `make local` | Run pipeline locally (using Makefile) |
| `make azure` | Run on Azure ML (using Makefile) |
| `make clean` | Remove output files |

## Next Steps

1. **Explore the code**: Read [README.md](README.md) for architecture overview
2. **Modify data**: Edit `config.py` to use your own CSV files
3. **Add models**: Extend `train.py` with new algorithm implementations
4. **Customize metrics**: Modify `evaluate.py` for domain-specific evaluation
5. **Deploy to production**: Follow guidance in [README.md](README.md#deployment)

## Additional Resources

- [Complete Documentation](README.md)
- [Azure ML Documentation](https://learn.microsoft.com/azure/machine-learning)
- [Scikit-learn Guide](https://scikit-learn.org/stable/user_guide.html)
- [pandas Documentation](https://pandas.pydata.org/docs/)

## Getting Help

Check the [Troubleshooting section in README.md](README.md#troubleshooting) for detailed solutions.

---

**Happy ML Ops! 🚀**
