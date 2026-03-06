# Local Inference Guide

This guide explains how to use the local inference script to make predictions with your trained Linear Regression model.

## Quick Start

### 1. Train a Model First

Before running inference, you need a trained model:

```bash
make local
```

This will train a model and save it to the `models/` directory.

### 2. Run Inference

#### Interactive Mode (Recommended for Testing)

```bash
make infer
# or
python inference.py --interactive
```

You'll be prompted to enter feature values:
```
Enter 10 feature values (comma-separated):
> 1.2, 3.4, 5.6, 7.8, 9.0, 2.1, 4.3, 6.5, 8.7, 0.9

==================================================
Input: [1.2, 3.4, 5.6, 7.8, 9.0, 2.1, 4.3, 6.5, 8.7, 0.9]
Prediction: 42.1234
==================================================
```

## Usage Modes

### 1. Interactive Mode

Enter feature values manually through prompts:

```bash
python inference.py --interactive
```

### 2. CSV Batch Prediction

Predict on multiple samples from a CSV file:

```bash
# Basic CSV prediction
python inference.py --csv data/test_data.csv

# Save predictions to file
python inference.py --csv data/test_data.csv --output predictions.csv

# Using Makefile
make infer-csv CSV=data/test_data.csv
make infer-csv CSV=data/test_data.csv OUTPUT=predictions.csv
```

**CSV Format:**
```csv
feature_0,feature_1,feature_2,...,feature_9
1.23,4.56,7.89,...,0.12
2.34,5.67,8.90,...,1.23
```

If the CSV includes a `target` column, the script will automatically:
- Calculate prediction metrics (MSE, RMSE, MAE, R²)
- Add error columns to output

### 3. JSON Prediction

Predict from JSON data:

```bash
# From JSON file
python inference.py --json data/input.json

# Save results
python inference.py --json data/input.json --output results.json
```

**JSON Format Options:**

Option 1 - Array of samples:
```json
{
  "features": [
    [1.2, 3.4, 5.6, 7.8, 9.0, 2.1, 4.3, 6.5, 8.7, 0.9],
    [2.3, 4.5, 6.7, 8.9, 1.0, 3.2, 5.4, 7.6, 9.8, 1.2]
  ]
}
```

Option 2 - Simple list:
```json
[[1.2, 3.4, 5.6, 7.8, 9.0, 2.1, 4.3, 6.5, 8.7, 0.9]]
```

Option 3 - Named features (single sample):
```json
{
  "feature_0": 1.2,
  "feature_1": 3.4,
  "feature_2": 5.6
}
```

### 4. Single Command-Line Prediction

Quick prediction from command line:

```bash
python inference.py --features 1.2,3.4,5.6,7.8,9.0,2.1,4.3,6.5,8.7,0.9
```

### 5. Using Specific Model

By default, the script uses the latest model. To use a specific model:

```bash
python inference.py --model models/my_specific_model.pkl --csv data/test.csv
```

## Examples

### Example 1: Quick Test

```bash
# Train model
make local

# Test with interactive mode
make infer
```

### Example 2: Batch Processing

```bash
# Create test CSV
cat > test_data.csv << EOF
feature_0,feature_1,feature_2,feature_3,feature_4,feature_5,feature_6,feature_7,feature_8,feature_9
1.2,3.4,5.6,7.8,9.0,2.1,4.3,6.5,8.7,0.9
2.3,4.5,6.7,8.9,1.0,3.2,5.4,7.6,9.8,1.2
EOF

# Run predictions
python inference.py --csv test_data.csv --output predictions.csv
```

### Example 3: Production API Integration

```python
from inference import ModelInference

# Initialize inference handler
inference = ModelInference()

# Make prediction
features = [1.2, 3.4, 5.6, 7.8, 9.0, 2.1, 4.3, 6.5, 8.7, 0.9]
prediction = inference.predict_single(features)
print(f"Prediction: {prediction}")

# Batch prediction
import numpy as np
X = np.array([
    [1.2, 3.4, 5.6, 7.8, 9.0, 2.1, 4.3, 6.5, 8.7, 0.9],
    [2.3, 4.5, 6.7, 8.9, 1.0, 3.2, 5.4, 7.6, 9.8, 1.2]
])
predictions = inference.predict(X)
print(predictions)
```

## Output Format

### CSV Output

When predicting from CSV with a target column:

```csv
feature_0,feature_1,...,prediction,target,error,abs_error
1.23,4.56,...,42.12,42.50,-0.38,0.38
2.34,5.67,...,53.21,53.00,0.21,0.21
```

### JSON Output

```json
{
  "predictions": [42.1234, 53.2145],
  "num_samples": 2,
  "timestamp": "2026-03-06T10:30:45.123456"
}
```

## Troubleshooting

### No Model Found

**Error:** `FileNotFoundError: No model files found in models/`

**Solution:** Train a model first:
```bash
make local
```

### Wrong Number of Features

**Error:** `Expected 10 features, got 5`

**Solution:** Ensure your input has the same number of features the model was trained with. Check model info:
```bash
python inference.py --interactive
# Look for: "Model expects X features"
```

### Model Loading Error

**Error:** `Failed to load model`

**Solution:** Ensure the model file is not corrupted. Retrain if necessary:
```bash
make clean
make local
```

## Integration with Applications

### REST API Example

```python
from flask import Flask, request, jsonify
from inference import ModelInference

app = Flask(__name__)
inference = ModelInference()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = data.get('features')

    if not features:
        return jsonify({'error': 'Missing features'}), 400

    try:
        prediction = inference.predict_single(features)
        return jsonify({
            'prediction': float(prediction),
            'model': str(inference.model_path.name)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### FastAPI Example

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from inference import ModelInference

app = FastAPI()
inference = ModelInference()

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: float
    model: str

@app.post('/predict', response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        prediction = inference.predict_single(request.features)
        return PredictionResponse(
            prediction=float(prediction),
            model=str(inference.model_path.name)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Performance Tips

1. **Model Loading:** The model is loaded once during initialization. For production, keep the `ModelInference` instance alive.

2. **Batch Predictions:** Use `predict()` for multiple samples rather than calling `predict_single()` in a loop:
   ```python
   # Good - batch prediction
   predictions = inference.predict(X_batch)

   # Bad - loop
   predictions = [inference.predict_single(x) for x in X_batch]
   ```

3. **CSV Processing:** For large CSV files, consider chunking:
   ```python
   import pandas as pd

   for chunk in pd.read_csv('large_file.csv', chunksize=1000):
       X = chunk.values
       predictions = inference.predict(X)
       # Process predictions
   ```

## Next Steps

- Check [QUICKSTART.md](QUICKSTART.md) for pipeline overview
- See [demo.py](demo.py) for advanced usage examples
- Read [README.md](README.md) for full project documentation
