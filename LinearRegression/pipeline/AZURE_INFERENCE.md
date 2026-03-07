# Azure ML Inference Guide

This guide explains how to deploy your trained Linear Regression model to Azure ML and make predictions using managed online endpoints.

## Overview

Azure ML provides **Managed Online Endpoints** for real-time inference with:
- **Auto-scaling**: Automatic scaling based on load
- **Monitoring**: Built-in metrics and logging
- **Security**: Key-based or token-based authentication
- **High availability**: 99.9% SLA
- **Versioning**: Blue/green deployments

## Quick Start

### 1. Train a Model Locally

```bash
make local
```

This creates a model file in the `models/` directory.

### 2. Deploy to Azure ML

```bash
make deploy-azure
```

This will:
- Register your model in Azure ML
- Create a managed online endpoint
- Deploy the model (takes ~10-15 minutes)
- Save endpoint configuration to `outputs/endpoint_config.json`

### 3. Test the Endpoint

```bash
make test-azure
```

This sends a sample request to verify the endpoint is working.

### 4. Make Predictions

```bash
# Interactive mode
make infer-azure

# Batch prediction from CSV
make infer-azure CSV=data/test_sample.csv

# Single prediction
make infer-azure FEATURES="0.7,-1.5,-0.2,0.6,-0.01,-0.4,0.07,1.1,-0.07,0.6"
```

## Detailed Usage

### Deployment

#### Basic Deployment

```bash
python deploy_azure.py
```

This uses the latest trained model and default endpoint name.

#### Deploy Specific Model

```bash
python deploy_azure.py --model models/my_model.pkl
```

#### Custom Endpoint Name

```bash
python deploy_azure.py --endpoint my-custom-endpoint
```

#### Deployment Output

After successful deployment, you'll see:

```
==================================================================
AZURE ML ENDPOINT DETAILS
==================================================================
Endpoint Name:    linear-regression-lr-endpoint
Scoring URI:      https://xxxxx.azureml.ms/score
Swagger URI:      https://xxxxx.azureml.ms/swagger.json
Auth Mode:        key
Provisioning:     Succeeded

Primary Key:
  xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

Secondary Key:
  xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
==================================================================
```

This information is saved to `outputs/endpoint_config.json` for future use.

### Making Predictions

#### Test Endpoint

Quick test with sample data:

```bash
python inference_azure.py --test
```

#### Interactive Mode

```bash
python inference_azure.py --interactive
```

You'll be prompted to enter feature values:
```
Enter 10 feature values (comma-separated):
> 0.714,-1.515,-0.223,0.617,-0.015,-0.366,0.071,1.095,-0.073,0.626

==================================================
Input: [0.714, -1.515, -0.223, 0.617, -0.015, -0.366, 0.071, 1.095, -0.073, 0.626]
Prediction: -215.1422
==================================================
```

#### CSV Batch Prediction

```bash
# Basic
python inference_azure.py --csv data/test_sample.csv

# Save results
python inference_azure.py --csv data/test_sample.csv --output azure_predictions.csv
```

If the CSV includes a `target` column, metrics will be calculated:
```
==================================================
AZURE ENDPOINT PREDICTION METRICS
==================================================
MSE: 102.3456
RMSE: 10.1165
MAE: 8.4532
R² Score: 0.9876
==================================================
```

#### Single Command-Line Prediction

```bash
python inference_azure.py --features "0.7,-1.5,-0.2,0.6,-0.01,-0.4,0.07,1.1,-0.07,0.6"
```

#### Using Specific Endpoint

```bash
# Load from saved config
python inference_azure.py --endpoint my-endpoint --test

# Provide credentials directly
python inference_azure.py \
  --scoring-uri "https://xxxxx.azureml.ms/score" \
  --api-key "your-key-here" \
  --test
```

## Programmatic Usage

### Python Client Example

```python
from inference_azure import AzureMLInference

# Initialize client (loads from outputs/endpoint_config.json)
client = AzureMLInference()

# Single prediction
features = [0.714, -1.515, -0.223, 0.617, -0.015,
            -0.366, 0.071, 1.095, -0.073, 0.626]
prediction = client.predict_single(features)
print(f"Prediction: {prediction}")

# Batch prediction
import numpy as np
X = np.array([
    [0.714, -1.515, -0.223, 0.617, -0.015, -0.366, 0.071, 1.095, -0.073, 0.626],
    [-0.234, 0.543, -0.234, -0.047, 0.028, 1.365, -0.457, 1.004, 0.767, 0.497]
])
result = client.predict(X)
print(result['predictions'])
```

### Custom Client with Credentials

```python
from inference_azure import AzureMLInference

client = AzureMLInference(
    scoring_uri="https://xxxxx.azureml.ms/score",
    api_key="your-key-here"
)

prediction = client.predict_single(features)
```

### CSV Processing

```python
from inference_azure import AzureMLInference

client = AzureMLInference()
results_df = client.predict_from_csv(
    'data/test_sample.csv',
    output_path='azure_predictions.csv'
)

print(results_df.head())
```

## API Request Format

### Input Format

The endpoint accepts JSON with the following formats:

**Format 1 - Features array:**
```json
{
  "features": [
    [0.714, -1.515, -0.223, 0.617, -0.015, -0.366, 0.071, 1.095, -0.073, 0.626],
    [-0.234, 0.543, -0.234, -0.047, 0.028, 1.365, -0.457, 1.004, 0.767, 0.497]
  ]
}
```

**Format 2 - Data array:**
```json
{
  "data": [
    [0.714, -1.515, -0.223, 0.617, -0.015, -0.366, 0.071, 1.095, -0.073, 0.626]
  ]
}
```

**Format 3 - Simple array:**
```json
[
  [0.714, -1.515, -0.223, 0.617, -0.015, -0.366, 0.071, 1.095, -0.073, 0.626]
]
```

### Response Format

```json
{
  "predictions": [-215.1422, 374.1605],
  "model_type": "LinearRegression",
  "num_samples": 2,
  "model_info": {
    "num_features": 10,
    "has_intercept": true
  }
}
```

## Direct HTTP Requests

### Using cURL

```bash
# Load endpoint info
SCORING_URI=$(jq -r '.scoring_uri' outputs/endpoint_config.json)
API_KEY=$(jq -r '.primary_key' outputs/endpoint_config.json)

# Make request
curl -X POST "$SCORING_URI" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "features": [
      [0.714, -1.515, -0.223, 0.617, -0.015, -0.366, 0.071, 1.095, -0.073, 0.626]
    ]
  }'
```

### Using Python Requests

```python
import json
import requests

# Load config
with open('outputs/endpoint_config.json') as f:
    config = json.load(f)

# Prepare request
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {config["primary_key"]}'
}

data = {
    "features": [
        [0.714, -1.515, -0.223, 0.617, -0.015, -0.366, 0.071, 1.095, -0.073, 0.626]
    ]
}

# Make request
response = requests.post(
    config['scoring_uri'],
    json=data,
    headers=headers
)

result = response.json()
print(result['predictions'])
```

### Using JavaScript/Node.js

```javascript
const axios = require('axios');
const fs = require('fs');

// Load config
const config = JSON.parse(fs.readFileSync('outputs/endpoint_config.json'));

// Make request
axios.post(config.scoring_uri, {
  features: [
    [0.714, -1.515, -0.223, 0.617, -0.015, -0.366, 0.071, 1.095, -0.073, 0.626]
  ]
}, {
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${config.primary_key}`
  }
})
.then(response => {
  console.log(response.data.predictions);
})
.catch(error => {
  console.error('Error:', error.response.data);
});
```

## Endpoint Management

### List Endpoints

```bash
az ml online-endpoint list \
  --resource-group <resource-group> \
  --workspace-name <workspace>
```

### Get Endpoint Details

```bash
az ml online-endpoint show \
  --name <endpoint-name> \
  --resource-group <resource-group> \
  --workspace-name <workspace>
```

### Update Endpoint

```bash
# Scale up instances
az ml online-deployment update \
  --name blue \
  --endpoint-name <endpoint-name> \
  --instance-count 3 \
  --resource-group <resource-group> \
  --workspace-name <workspace>
```

### Delete Endpoint

```bash
az ml online-endpoint delete \
  --name <endpoint-name> \
  --resource-group <resource-group> \
  --workspace-name <workspace> \
  --yes
```

## Monitoring & Troubleshooting

### View Endpoint Metrics

In Azure Portal:
1. Navigate to your Azure ML workspace
2. Go to **Endpoints** → **Real-time endpoints**
3. Click on your endpoint
4. View **Metrics** tab for:
   - Request latency
   - Request rate
   - Success rate
   - CPU/Memory usage

### View Logs

```bash
az ml online-deployment get-logs \
  --name blue \
  --endpoint-name <endpoint-name> \
  --resource-group <resource-group> \
  --workspace-name <workspace> \
  --lines 100
```

### Common Issues

#### 1. Deployment Takes Too Long

**Cause:** Azure is provisioning compute resources

**Solution:** Wait 10-15 minutes. Check status:
```bash
az ml online-endpoint show --name <endpoint-name> \
  --resource-group <resource-group> \
  --workspace-name <workspace>
```

#### 2. "Model not found" Error

**Cause:** Model file missing or incorrect path

**Solution:** Ensure model is trained and registered:
```bash
make local  # Train model first
```

#### 3. Authentication Failed

**Cause:** Invalid or expired API key

**Solution:** Regenerate key:
```bash
az ml online-endpoint regenerate-keys \
  --name <endpoint-name> \
  --resource-group <resource-group> \
  --workspace-name <workspace> \
  --key-type primary
```

#### 4. Prediction Error

**Cause:** Wrong input format or feature count

**Solution:** Verify input has exactly 10 features:
```bash
python inference_azure.py --test
```

## Cost Optimization

### Estimating Costs

Endpoint costs depend on:
- **Instance type**: Standard_DS2_v2 (~$0.14/hour)
- **Number of instances**: Default is 1
- **Running time**: Pay per hour

**Example cost:**
- 1 instance × 24 hours × 30 days = ~$100/month

### Reduce Costs

1. **Delete when not in use:**
   ```bash
   az ml online-endpoint delete --name <endpoint-name> --yes
   ```

2. **Scale down instances:**
   ```bash
   az ml online-deployment update --instance-count 1
   ```

3. **Use smaller instance type:**
   Edit `deploy_azure.py` and change `instance_type` to `Standard_DS1_v2`

4. **Use batch endpoints** for non-real-time predictions (cheaper)

## Best Practices

### 1. Environment Variables

Store credentials in environment variables:

```bash
export SCORING_URI="https://xxxxx.azureml.ms/score"
export API_KEY="your-key-here"
```

### 2. Error Handling

```python
from inference_azure import AzureMLInference
import logging

try:
    client = AzureMLInference()
    result = client.predict(data)
except Exception as e:
    logging.error(f"Prediction failed: {e}")
    # Handle error appropriately
```

### 3. Retry Logic

```python
import time
from requests.exceptions import RequestException

max_retries = 3
for attempt in range(max_retries):
    try:
        result = client.predict(data)
        break
    except RequestException as e:
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # Exponential backoff
            continue
        raise
```

### 4. Batch Processing

For large datasets, process in batches:

```python
import pandas as pd

chunk_size = 100
df = pd.read_csv('large_file.csv')

all_predictions = []
for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i:i+chunk_size]
    X = chunk.drop(columns=['target']).values
    result = client.predict(X)
    all_predictions.extend(result['predictions'])
```

## Security Considerations

1. **Never commit API keys** to version control
2. **Use Azure Key Vault** for production
3. **Rotate keys regularly**
4. **Use managed identities** when possible
5. **Enable network isolation** for sensitive workloads

## Next Steps

- **Monitor endpoint performance** in Azure Portal
- **Set up alerts** for high latency or errors
- **Implement A/B testing** with blue/green deployments
- **Add authentication** to your application
- **Optimize model** for faster inference

## Related Documentation

- [INFERENCE.md](INFERENCE.md) - Local inference guide
- [QUICKSTART.md](QUICKSTART.md) - Pipeline overview
- [README.md](README.md) - Full project documentation
- [Azure ML Endpoints Documentation](https://learn.microsoft.com/azure/machine-learning/concept-endpoints)
