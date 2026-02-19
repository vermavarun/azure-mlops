# 📚 MLOps Pipeline - Complete Roadmap

Welcome to the Azure MLOps Linear Regression Pipeline! This guide helps you navigate everything in the project.

## 🚀 Start Here

Choose your path based on your goals:

### ⏱️ **I have 5 minutes** (Quick Test)
1. Read: [QUICKSTART.md](QUICKSTART.md)
2. Run: `python demo.py`
3. Explore generated files in `metrics/` and `models/`

### ⏱️ **I have 30 minutes** (Full Setup)
1. Complete: [QUICKSTART.md](QUICKSTART.md)
2. Configure: Copy `.env.example` → `.env`, update values
3. Run: `python app.py` (local) or `python app.py --azure` (cloud)
4. Review: Check outputs in `metrics/`

### ⏱️ **I have 1 hour** (Deep Dive)
1. Read: [README.md](README.md) (complete architecture)
2. Read: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) (file reference)
3. Study: `demo.py` for code examples
4. Run: `python run_experiments.py --experiment all`
5. Explore: Experiment results in `experiments/`

### ⏱️ **I'm a Developer** (Contribution Ready)
1. Read: [CONTRIBUTING.md](CONTRIBUTING.md)
2. Setup: `pip install -r requirements-dev.txt`
3. Code: Make changes in core modules
4. Test: `make format && make lint && make test`
5. Review: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for architecture

---

## 📂 File Organization by Purpose

### 🔧 **Core Pipeline** (What you'll use)
| File | Purpose | When to use |
|------|---------|------------|
| [app.py](app.py) | Main entry point | `python app.py` |
| [config.py](config.py) | Configuration loader | Modify settings |
| [data_handler.py](data_handler.py) | Data operations | Load/prepare data |
| [train.py](train.py) | Model training | Train new models |
| [evaluate.py](evaluate.py) | Evaluation & metrics | Assess performance |

### 📖 **Documentation** (What to read)
| File | Purpose | Read when |
|------|---------|-----------|
| [README.md](README.md) | Complete guide | First time setup |
| [QUICKSTART.md](QUICKSTART.md) | 5-minute guide | Want quick run |
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | File reference | Need to find something |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute | Planning code changes |

### 🔬 **Examples & Experiments** (What to run)
| File | Purpose | Run with |
|------|---------|----------|
| [demo.py](demo.py) | 5 workflow examples | `python demo.py` |
| [run_experiments.py](run_experiments.py) | Systematic testing | `python run_experiments.py` |
| [experiment_configs.py](experiment_configs.py) | Pre-made configs | Import in your code |

### ⚙️ **Automation** (How to simplify)
| File | Platform | Commands |
|------|----------|----------|
| [Makefile](Makefile) | Unix/macOS | `make help` |
| [pipeline.sh](pipeline.sh) | Linux/macOS | `./pipeline.sh local` |
| [pipeline.ps1](pipeline.ps1) | Windows | `.\pipeline.ps1 -command local` |

### 📦 **Dependencies** (What's installed)
| File | Purpose | Install with |
|------|---------|--------------|
| [requirements.txt](requirements.txt) | Core packages | `pip install -r requirements.txt` |
| [requirements-dev.txt](requirements-dev.txt) | Dev tools | `pip install -r requirements-dev.txt` |

### 🐳 **Containerization** (Docker)
| File | Purpose | Use when |
|------|---------|----------|
| [Dockerfile](Dockerfile) | Container image | Want to containerize |
| [docker-compose.yml](docker-compose.yml) | Multi-service setup | Need Jupyter + Pipeline |

### 🔄 **CI/CD** (Automation)
| File | Purpose | Setup at |
|------|---------|----------|
| [.github/workflows/ci-cd.yml](.github/workflows/ci-cd.yml) | Auto testing | GitHub repository |
| [.github/workflows/deploy.yml](.github/workflows/deploy.yml) | Manual deployment | GitHub Actions |

### 🎨 **VS Code** (IDE Configuration)
| File | Purpose | Enabled by |
|------|---------|-----------|
| [.vscode/launch.json](.vscode/launch.json) | Debug configs | Opening in VS Code |
| [.vscode/settings.json](.vscode/settings.json) | Code formatting | VS Code Pylance |
| [.vscode/extensions.json](.vscode/extensions.json) | Recommendations | VS Code prompt |

### ⚙️ **Configuration** (Settings)
| File | Purpose | Update |
|------|---------|--------|
| [.env.example](.env.example) | Config template | Copy to `.env` |
| [.gitignore](.gitignore) | Git exclusions | Usually no changes |

---

## 🎯 Common Tasks

### Task: Run pipeline locally with synthetic data
```bash
cp .env.example .env      # One-time setup
python app.py             # Run it
```
📁 Output: `models/`, `metrics/`  
📖 See: [QUICKSTART.md](QUICKSTART.md)

### Task: Train with your own data
```bash
# 1. Update config.py
# 2. Set: use_synthetic = False
# 3. Provide CSV file path
python app.py
```
📖 See: [README.md](README.md#using-your-own-data)

### Task: Compare 4 regre models
```bash
python run_experiments.py --experiment comparison
```
📁 Output: `experiments/YYYYMMDD_HHMMSS/`  
📖 See: [run_experiments.py](run_experiments.py)

### Task: Deploy to Azure ML
```bash
# 1. Configure Azure credentials in .env
# 2. Run:
python app.py --azure
```
📖 See: [README.md](README.md#azure-ml-integration)

### Task: Run as Docker container
```bash
docker build -t mlops-lr .
docker run --env-file .env mlops-lr
```
📖 See: [Dockerfile](Dockerfile)

### Task: Use Jupyter for exploration
```bash
docker-compose up jupyter
# Access at: http://localhost:8888
```
📖 See: [docker-compose.yml](docker-compose.yml)

### Task: Add a new regression model
1. Update: [train.py](train.py) - Add `train_my_model()` function
2. Update: [train.py](train.py) - Add to factory function
3. Update: [demo.py](demo.py) - Add demo
4. Update: [experiment_configs.py](experiment_configs.py) - Add config
📖 See: [CONTRIBUTING.md](CONTRIBUTING.md#new-models)

### Task: Format and lint code
```bash
make format    # Format with Black
make lint      # Check with Pylint
```
Or manually:
```bash
black . --line-length 100
pylint *.py
```
📖 See: [CONTRIBUTING.md](CONTRIBUTING.md#code-style)

### Task: Create a Git repository
```bash
git init
git add .
git config user.name "Your Name"
git config user.email "your@email.com"
git commit -m "Initial commit: MLOps pipeline"
git remote add origin https://github.com/YOUR/REPO.git
git branch -M main
git push -u origin main
```

### Task: Set up GitHub Actions CI/CD
1. Push to GitHub (from above)
2. Go to: Settings → Secrets and variables → Actions
3. Add secrets:
   - `AZURE_CREDENTIALS` (from `az ad sp create-for-rbac`)
   - `AZURE_SUBSCRIPTION_ID`
   - `AZURE_RESOURCE_GROUP`
   - `AZURE_WORKSPACE_NAME`
4. Workflows run on: push, pull request
📖 See: [.github/workflows/ci-cd.yml](.github/workflows/ci-cd.yml)

---

## 📊 Data Flow Diagram

```
Input Data          Processing           Output
─────────────────────────────────────────────────────────
┌─────────────────┐
│ Synthetic Data  │  ─┐
│ OR CSV File     │   │
└─────────────────┘   │
                      ├──→ data_handler.py
                      │    • Load data
                      │    • Split (train/test)
                      │    • Scale features
                      │    • Remove outliers
                      │
                      └──→ train.py ────────→ models/
                           • Train model      (trained_model.pkl)
                           • Fit parameters
                           
                      └──→ evaluate.py ──────→ metrics/
                           • Compute R², RMSE (metrics.json)
                           • Cross-validation  (plots.png)
                           • Generate plots
```

---

## 🐛 Troubleshooting Guide

### "ModuleNotFoundError: No module named 'X'"
Solution: `pip install -r requirements.txt`  
📖 See: [QUICKSTART.md#troubleshooting](QUICKSTART.md#troubleshooting)

### "Azure CLI not found"
Solution: `pip install azure-cli` then `az login`  
📖 See: [README.md#azure-authentication](README.md)

### ".env file not found"
Solution: `cp .env.example .env` then edit  
📖 See: [QUICKSTART.md#step-2-configure-azure](QUICKSTART.md#step-2-configure-azure)

### "DEPRECATED WARNING" messages
Solution: These are fine, features still work. Update packages occasionally.  
📖 See: [requirements.txt](requirements.txt)

### "Port 8888 already in use" (Docker)
Solution: `docker-compose down` or change port in docker-compose.yml  
📖 See: [docker-compose.yml](docker-compose.yml)

---

## 📚 Learning Path

### Beginner (Learn the pipeline)
1. [QUICKSTART.md](QUICKSTART.md) - Get it running
2. [demo.py](demo.py) - See examples
3. [README.md](README.md#architecture) - Understand structure

### Intermediate (Use for your data)
1. [README.md](README.md#using-your-own-data) - Load your data
2. [config.py](config.py) - Customize settings
3. [run_experiments.py](run_experiments.py) - Run systematic tests
4. [experiment_configs.py](experiment_configs.py) - Create configs

### Advanced (Extend & deploy)
1. [CONTRIBUTING.md](CONTRIBUTING.md) - Development setup
2. [train.py](train.py) - Add new models
3. [Docker](Dockerfile) - Containerize
4. [CI/CD](.github/workflows/ci-cd.yml) - Automate

### Expert (Production deployment)
1. [README.md](README.md#deployment) - Production setup
2. [Azure ML Integration](README.md#azure-ml-integration)
3. [Monitoring](README.md#monitoring-and-logging)
4. [Model Registry](README.md#model-registration)

---

## 🎓 Key Concepts

### Models Included
- **Linear Regression**: Simple, interpretable baseline
- **Polynomial Regression**: Capture nonlinear relationships
- **Ridge Regression**: L2 regularization, reduce overfitting
- **Lasso Regression**: L1 regularization, feature selection

### Metrics Computed
- **R² Score**: Proportion of variance explained (0-1, higher better)
- **RMSE**: Root Mean Squared Error (lower better)
- **MAE**: Mean Absolute Error (lower better)
- **MAPE**: Mean Absolute Percentage Error (lower better)

### File Formats
- **Models**: Joblib `.pkl` (pickled Python objects)
- **Data**: NumPy `.npy` (binary array format)
- **Metrics**: JSON `.json` (human + machine readable)
- **Plots**: PNG `.png` (images, versioned)

---

## 🔗 Important Files at a Glance

**Start here**:
- 📄 [QUICKSTART.md](QUICKSTART.md)
- 📄 [README.md](README.md)

**Do this first**:
- 📄 [requirements.txt](requirements.txt) - Install deps
- 📄 [.env.example](.env.example) - Configure Azure
- 🐍 [demo.py](demo.py) - Run examples

**Understand the code**:
- 🐍 [app.py](app.py) - Entry point
- 🐍 [train.py](train.py) - Models
- 🐍 [evaluate.py](evaluate.py) - Metrics
- 📄 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - File guide

**Extend it**:
- 📄 [CONTRIBUTING.md](CONTRIBUTING.md) - How to modify
- 🐍 [experiment_configs.py](experiment_configs.py) - Configs
- 🐍 [run_experiments.py](run_experiments.py) - Batch testing

**Deploy it**:
- 📄 [Dockerfile](Dockerfile) - Containerize
- 📄 [docker-compose.yml](docker-compose.yml) - Orchestrate
- 📄 [.github/workflows/ci-cd.yml](.github/workflows/ci-cd.yml) - Automate

---

## ✅ Verification Checklist

After setup, verify everything works:

- [ ] `python -c "import app; print('✓ app imports')"` 
- [ ] `python demo.py` (runs without errors)
- [ ] `ls models/` (has trained_model.pkl files)
- [ ] `ls metrics/` (has .json and .png files)
- [ ] `python app.py --azure` (submits to Azure ML)
- [ ] `docker build -t mlops-lr .` (builds successfully)

---

## 🆘 Getting Help

### For usage questions
→ Check [README.md](README.md#troubleshooting)

### For setup issues
→ Check [QUICKSTART.md#troubleshooting](QUICKSTART.md#troubleshooting)

### For code modifications
→ Read [CONTRIBUTING.md](CONTRIBUTING.md)

### For infrastructure
→ Check `../Infra/` directory (Terraform)

### For Azure ML specifics
→ Visit [Azure ML Documentation](https://learn.microsoft.com/azure/machine-learning)

---

## 📞 Quick Reference

| Need | Do this | See |
|------|---------|-----|
| Run locally | `python app.py` | [QUICKSTART.md](QUICKSTART.md) |
| Run examples | `python demo.py` | [demo.py](demo.py) |
| Deploy to cloud | `python app.py --azure` | [README.md](README.md) |
| Configure | Edit `.env` | [.env.example](.env.example) |
| Install deps | `pip install -r requirements.txt` | [requirements.txt](requirements.txt) |
| Format code | `make format` | [Makefile](Makefile) |
| Run tests | `make test` | [Makefile](Makefile) |
| Docker build | `docker build -t mlops-lr .` | [Dockerfile](Dockerfile) |
| Docker run | `docker run --env-file .env mlops-lr` | [docker-compose.yml](docker-compose.yml) |

---

**Last Updated**: 2024  
**Status**: ✅ Production Ready  
**License**: MIT  

🎉 **Happy ML Ops!**
