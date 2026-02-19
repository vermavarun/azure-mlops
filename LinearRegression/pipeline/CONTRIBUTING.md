# Contributing to MLOps Linear Regression Pipeline

Thank you for your interest in contributing to this project! Here's how to help:

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/azure-mlops.git
   cd azure-mlops/LinearRegression/pipeline
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install development dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   ```

## Development Workflow

### Code Style

- **Format code** with Black (100 character line length):
  ```bash
  black . --line-length 100
  ```
- **Lint code** with Pylint:
  ```bash
  pylint --disable=all --enable=E,F *.py
  ```
- **Check imports** with isort:
  ```bash
  isort .
  ```

### Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Keep commits small and focused
   - Write descriptive commit messages
   - Add docstrings to new functions/classes

3. **Test your changes**:
   ```bash
   # Run local pipeline
   python app.py
   
   # Run demo script
   python demo.py
   
   # Test specific module
   python -m pytest tests/ -v
   ```

4. **Format and lint**:
   ```bash
   make format
   make lint
   ```

## Types of Contributions

### Bug Fixes
- Open an issue describing the bug
- Fork and create a fix
- Include a test demonstrating the fix
- Submit a pull request

### New Features
- Discuss the feature in an issue first
- Implement with clear, documented code
- Add examples in `demo.py`
- Update `README.md` with usage

### Documentation
- Improve clarity in existing documentation
- Add examples to QUICKSTART.md or README.md
- Add docstrings to functions
- Create how-to guides for common tasks

### New Models
To add a new regression model:

1. **Add training function** in `train.py`:
   ```python
   def train_my_model(X_train, y_train, **kwargs):
       """Train your custom model."""
       model = MyModel(**kwargs)
       model.fit(X_train, y_train)
       return model, model.score(X_train, y_train)
   ```

2. **Update `train_model()` factory**:
   ```python
   elif model_type == 'my_model':
       return train_my_model(X_train, y_train, **model_kwargs)
   ```

3. **Add to demo**:
   ```python
   # In demo.py
   for model_type in ['linear', 'ridge', 'lasso', 'my_model']:
       # ... add comparison
   ```

4. **Update config**:
   ```python
   # In config.py
   'model_type': os.getenv('MODEL_TYPE', 'my_model'),
   ```

### New Metrics
To add evaluation metrics:

1. **Add metric function** in `evaluate.py`:
   ```python
   def custom_metric(y_true, y_pred):
       """Compute custom metric."""
       return your_calculation
   ```

2. **Update `evaluate_model()`**:
   ```python
   metrics['custom_metric'] = custom_metric(y_test, y_pred)
   ```

3. **Update JSON schema** in `save_metrics()` as needed

## Pull Request Process

1. **Before submitting**:
   - Run `make format` and `make lint`
   - Test with `python app.py`
   - Test with `python demo.py`
   - Document your changes in docstrings

2. **Submit PR with**:
   - Clear title and description
   - Reference related issues (#123)
   - Screenshots for UI changes
   - Explanation of changes and testing

3. **PR Guidelines**:
   - Keep PRs focused on single feature/fix
   - Include descriptive commit history
   - Request review from maintainers
   - Address review feedback

## Code Review Expectations

- Code should be readable and maintainable
- Follow existing code patterns
- Include docstrings for new functions
- Handle edge cases (empty data, NaN values, etc.)
- Log important operations
- Test before submitting

## Testing

### Unit Tests
```bash
pytest tests/ -v
```

### Coverage Report
```bash
pytest --cov=. --cov-report=html
open htmlcov/index.html
```

### Integration Tests
```bash
python app.py
python app.py --azure
python demo.py
```

## Documentation

- **Update README.md** for new features
- **Update QUICKSTART.md** for usage changes
- **Add docstrings**: Use Google-style format
  ```python
  def function(param1, param2):
      """Short description.
      
      Longer description if needed.
      
      Args:
          param1: Description
          param2: Description
          
      Returns:
          Description of return value
          
      Raises:
          ValueError: When something is wrong
      """
  ```

## Azure ML Testing

To test with actual Azure ML:

1. **Configure credentials**:
   ```bash
   az login
   az account set --subscription YOUR_SUBSCRIPTION_ID
   ```

2. **Update .env** with actual workspace/resource details

3. **Test Azure path**:
   ```bash
   python app.py --azure
   ```

## Common Issues

### "Module import error"
```bash
pip install -e .  # Install in development mode
```

### "Azure authentication fails"
```bash
az logout
az login --use-device-code
```

### "Data directory not found"
```bash
mkdir -p data models metrics logs
```

## Questions?

- Open an issue with the `question` label
- Check existing issues and discussions
- Review documentation and examples in QUICKSTART.md

## License

By contributing, you agree your code follows the project license.

---

**Thank you for helping improve this MLOps pipeline! 🚀**
