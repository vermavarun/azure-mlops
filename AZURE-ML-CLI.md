# Azure ML CLI Commands Reference

Complete guide for Azure ML operations organized by workflow stages.

---

## 🔧 Setup & Configuration

### Login & Authentication
```bash
# Login to Azure
az login

# Set active subscription
az account set --subscription <subscription-id>

# Verify current subscription
az account show

# Login with service principal
az login --service-principal \
  --username <app-id> \
  --password <password-or-cert> \
  --tenant <tenant-id>
```

### Workspace Management
```bash
# Create resource group
az group create --name <resource-group> --location <location>

# Create Azure ML workspace
az ml workspace create \
  --name <workspace-name> \
  --resource-group <resource-group> \
  --location <location>

# Show workspace details
az ml workspace show \
  --name <workspace-name> \
  --resource-group <resource-group>

# List all workspaces
az ml workspace list --resource-group <resource-group>

# Update workspace
az ml workspace update \
  --name <workspace-name> \
  --resource-group <resource-group> \
  --description "Updated description"

# Delete workspace
az ml workspace delete \
  --name <workspace-name> \
  --resource-group <resource-group>
```

### Compute Resources
```bash
# Create compute cluster (for training)
az ml compute create \
  --name <compute-name> \
  --type AmlCompute \
  --min-instances 0 \
  --max-instances 4 \
  --size Standard_DS3_v2 \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>

# Show compute details
az ml compute show \
  --name <compute-name> \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>

# List all compute targets
az ml compute list \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>

# Update compute cluster (scale)
az ml compute update \
  --name <compute-name> \
  --min-instances 0 \
  --max-instances 8 \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>

# Delete compute cluster
az ml compute delete \
  --name <compute-name> \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>
```

### Environment Management
```bash
# Create environment from conda file
az ml environment create \
  --name linear-regression-env \
  --version 1 \
  --file conda.yml \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>

# Create environment from Docker
az ml environment create \
  --name custom-env \
  --docker-image mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04 \
  --conda-file conda.yml \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>

# List environments
az ml environment list \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>

# Show environment details
az ml environment show \
  --name linear-regression-env \
  --version 1 \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>
```

---

## 🚂 Training

### Data Management
```bash
# Create datastore
az ml datastore create \
  --name <datastore-name> \
  --type AzureBlob \
  --account-name <storage-account> \
  --container-name <container> \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>

# Upload data
az ml data create \
  --name training-data \
  --version 1 \
  --type uri_folder \
  --path ./data \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>

# List data assets
az ml data list \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>
```

### Submit Training Jobs
```bash
# Submit command job (single script)
az ml job create \
  --file job.yml \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>

# Submit job with inline command
az ml job create \
  --command "python train.py" \
  --code . \
  --environment linear-regression-env:1 \
  --compute-target <compute-name> \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>

# Submit pipeline job
az ml job create \
  --file pipeline.yml \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>

# Submit with parameters
az ml job create \
  --file job.yml \
  --set inputs.learning_rate=0.01 \
  --set inputs.batch_size=32 \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>
```

### Monitor Training Jobs
```bash
# List all jobs
az ml job list \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>

# Show job details
az ml job show \
  --name <job-name> \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>

# Stream job logs
az ml job stream \
  --name <job-name> \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>

# Download job outputs
az ml job download \
  --name <job-name> \
  --output-name model \
  --download-path ./outputs \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>

# Cancel running job
az ml job cancel \
  --name <job-name> \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>
```

### Experiment Tracking
```bash
# List experiments (via jobs)
az ml job list \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>

# Filter jobs by experiment
az ml job list \
  --query "[?tags.experiment=='linear-regression']" \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>
```

---

## 📦 Model Registration

### Register Model
```bash
# Register model from local path
az ml model create \
  --name linear-regression-model \
  --version 1 \
  --path ./models/linear_regression.pkl \
  --type mlflow_model \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>

# Register model from job output
az ml model create \
  --name linear-regression-model \
  --version 1 \
  --path azureml://jobs/<job-name>/outputs/model \
  --type custom_model \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>

# Register with description and tags
az ml model create \
  --name linear-regression-model \
  --version 1 \
  --path ./models/linear_regression.pkl \
  --description "Linear regression model for prediction" \
  --tags framework=sklearn algorithm=linear_regression \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>
```

### Manage Models
```bash
# List all models
az ml model list \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>

# Show model details
az ml model show \
  --name linear-regression-model \
  --version 1 \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>

# Download model
az ml model download \
  --name linear-regression-model \
  --version 1 \
  --download-path ./downloaded_models \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>

# Update model metadata
az ml model update \
  --name linear-regression-model \
  --version 1 \
  --set description="Updated model description" \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>

# Archive model (soft delete)
az ml model archive \
  --name linear-regression-model \
  --version 1 \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>
```

---

## 🚀 Deployment & Inference

### Online Endpoints (Real-time Inference)
```bash
# Create online endpoint
az ml online-endpoint create \
  --name linear-regression-endpoint \
  --auth-mode key \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>

# Create deployment
az ml online-deployment create \
  --name blue \
  --endpoint-name linear-regression-endpoint \
  --model linear-regression-model:1 \
  --instance-type Standard_DS3_v2 \
  --instance-count 1 \
  --scoring-script score.py \
  --environment linear-regression-env:1 \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>

# Update traffic allocation (blue-green deployment)
az ml online-endpoint update \
  --name linear-regression-endpoint \
  --traffic "blue=100" \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>

# List endpoints
az ml online-endpoint list \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>

# Show endpoint details
az ml online-endpoint show \
  --name linear-regression-endpoint \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>

# Get endpoint URI and credentials
az ml online-endpoint show \
  --name linear-regression-endpoint \
  --resource-group <resource-group> \
  --workspace-name <workspace-name> \
  --query scoring_uri

az ml online-endpoint get-credentials \
  --name linear-regression-endpoint \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>
```

### Batch Endpoints (Batch Inference)
```bash
# Create batch endpoint
az ml batch-endpoint create \
  --name linear-regression-batch \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>

# Create batch deployment
az ml batch-deployment create \
  --name default \
  --endpoint-name linear-regression-batch \
  --model linear-regression-model:1 \
  --compute <compute-name> \
  --instance-count 1 \
  --max-concurrency-per-instance 2 \
  --mini-batch-size 10 \
  --output-action append_row \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>

# Set default deployment
az ml batch-endpoint update \
  --name linear-regression-batch \
  --defaults deployment_name=default \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>

# Invoke batch endpoint
az ml batch-endpoint invoke \
  --name linear-regression-batch \
  --input azureml://datastores/workspaceblobstore/paths/inference-data \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>

# List batch jobs
az ml job list \
  --endpoint-name linear-regression-batch \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>
```

### Testing Endpoints
```bash
# Test online endpoint with sample data
az ml online-endpoint invoke \
  --name linear-regression-endpoint \
  --request-file sample_request.json \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>

# Test with curl
SCORING_URI=$(az ml online-endpoint show \
  --name linear-regression-endpoint \
  --resource-group <resource-group> \
  --workspace-name <workspace-name> \
  --query scoring_uri -o tsv)

API_KEY=$(az ml online-endpoint get-credentials \
  --name linear-regression-endpoint \
  --resource-group <resource-group> \
  --workspace-name <workspace-name> \
  --query primaryKey -o tsv)

curl -X POST $SCORING_URI \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

### Monitor Deployments
```bash
# Get deployment logs
az ml online-deployment get-logs \
  --name blue \
  --endpoint-name linear-regression-endpoint \
  --lines 100 \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>

# List deployment metrics
az ml online-deployment show \
  --name blue \
  --endpoint-name linear-regression-endpoint \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>
```

### Cleanup Deployments
```bash
# Delete deployment
az ml online-deployment delete \
  --name blue \
  --endpoint-name linear-regression-endpoint \
  --resource-group <resource-group> \
  --workspace-name <workspace-name> \
  --yes

# Delete endpoint
az ml online-endpoint delete \
  --name linear-regression-endpoint \
  --resource-group <resource-group> \
  --workspace-name <workspace-name> \
  --yes
```

---

## 📊 Monitoring & Logging

### Application Insights
```bash
# Get workspace Application Insights
az ml workspace show \
  --name <workspace-name> \
  --resource-group <resource-group> \
  --query application_insights -o tsv

# Query logs (requires App Insights extension)
az monitor app-insights query \
  --app <app-insights-id> \
  --analytics-query "traces | where message contains 'prediction' | take 100"
```

### Metrics
```bash
# View job metrics
az ml job show \
  --name <job-name> \
  --resource-group <resource-group> \
  --workspace-name <workspace-name> \
  --query metrics
```

---

## 🔄 Complete Workflow Examples

### End-to-End Training Workflow
```bash
# 1. Setup
az login
az account set --subscription <subscription-id>

# 2. Create/verify workspace
az ml workspace show -n <workspace> -g <resource-group>

# 3. Create compute
az ml compute create --name cpu-cluster --type AmlCompute \
  --min-instances 0 --max-instances 4 --size Standard_DS3_v2 \
  -g <resource-group> -w <workspace>

# 4. Create environment
az ml environment create --name lr-env --version 1 \
  --file conda.yml -g <resource-group> -w <workspace>

# 5. Submit training job
az ml job create --file job.yml -g <resource-group> -w <workspace>

# 6. Monitor job
az ml job stream --name <job-name> -g <resource-group> -w <workspace>

# 7. Register model from job
az ml model create --name lr-model --version 1 \
  --path azureml://jobs/<job-name>/outputs/model \
  -g <resource-group> -w <workspace>
```

### End-to-End Deployment Workflow
```bash
# 1. Create endpoint
az ml online-endpoint create --name lr-endpoint \
  -g <resource-group> -w <workspace>

# 2. Deploy model
az ml online-deployment create --name production \
  --endpoint-name lr-endpoint \
  --model lr-model:1 \
  --instance-type Standard_DS3_v2 \
  --instance-count 1 \
  -g <resource-group> -w <workspace>

# 3. Allocate traffic
az ml online-endpoint update --name lr-endpoint \
  --traffic "production=100" \
  -g <resource-group> -w <workspace>

# 4. Test endpoint
az ml online-endpoint invoke --name lr-endpoint \
  --request-file test_data.json \
  -g <resource-group> -w <workspace>

# 5. Monitor logs
az ml online-deployment get-logs --name production \
  --endpoint-name lr-endpoint --lines 100 \
  -g <resource-group> -w <workspace>
```

### Batch Inference Workflow
```bash
# 1. Create batch endpoint
az ml batch-endpoint create --name lr-batch \
  -g <resource-group> -w <workspace>

# 2. Create batch deployment
az ml batch-deployment create --name default \
  --endpoint-name lr-batch \
  --model lr-model:1 \
  --compute cpu-cluster \
  -g <resource-group> -w <workspace>

# 3. Invoke batch scoring
az ml batch-endpoint invoke --name lr-batch \
  --input azureml://datastores/workspaceblobstore/paths/batch-data \
  -g <resource-group> -w <workspace>

# 4. Check job status
az ml job list --endpoint-name lr-batch \
  -g <resource-group> -w <workspace>
```

---

## 🐍 Python SDK Equivalents (from app.py)

### Initialize ML Client
```python
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id="<subscription-id>",
    resource_group_name="<resource-group>",
    workspace_name="<workspace-name>"
)
```

### Create Environment
```python
from azure.ai.ml.entities import Environment

env = Environment(
    name="linear-regression-env",
    description="Environment for linear regression pipeline",
    conda_file="conda.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04"
)
ml_client.environments.create_or_update(env)
```

### Submit Job
```python
from azure.ai.ml import command

job = command(
    code=".",
    command="python train.py",
    environment="linear-regression-env:1",
    compute="cpu-cluster",
)
returned_job = ml_client.jobs.create_or_update(job)
```

### Register Model
```python
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

model = Model(
    path="./models/linear_regression.pkl",
    name="linear-regression-model",
    description="Linear regression model",
    type=AssetTypes.CUSTOM_MODEL,
)
ml_client.models.create_or_update(model)
```

### Deploy Model
```python
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    CodeConfiguration
)

# Create endpoint
endpoint = ManagedOnlineEndpoint(
    name="linear-regression-endpoint",
    auth_mode="key"
)
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# Create deployment
deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name="linear-regression-endpoint",
    model="linear-regression-model:1",
    environment="linear-regression-env:1",
    code_configuration=CodeConfiguration(
        code=".", scoring_script="score.py"
    ),
    instance_type="Standard_DS3_v2",
    instance_count=1
)
ml_client.online_deployments.begin_create_or_update(deployment).result()

# Allocate traffic
endpoint.traffic = {"blue": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
```

---

## 📝 Configuration Files

### job.yml (Training Job)
```yaml
$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
command: python train.py
code: .
environment: azureml:linear-regression-env:1
compute: azureml:cpu-cluster
experiment_name: linear-regression
description: Train linear regression model
```

### sample_request.json (Inference)
```json
{
  "input_data": {
    "columns": ["feature1", "feature2", "feature3"],
    "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
  }
}
```

---

## 🔗 Useful Links

- [Azure ML CLI Documentation](https://learn.microsoft.com/en-us/cli/azure/ml)
- [Azure ML Python SDK](https://learn.microsoft.com/en-us/python/api/azure-ai-ml/)
- [MLOps Best Practices](https://learn.microsoft.com/en-us/azure/machine-learning/concept-model-management-and-deployment)
- [Training Pipeline Examples](https://github.com/Azure/azureml-examples)

---

## 💡 Tips & Tricks

1. **Set defaults** to avoid repeating parameters:
   ```bash
   az configure --defaults workspace=<workspace> group=<resource-group>
   ```

2. **Use output formats**:
   ```bash
   az ml model list -o table  # table format
   az ml model show --name <model> --query name -o tsv  # get specific field
   ```

3. **Parallel jobs** for faster training:
   ```bash
   az ml job create --file job.yml --set compute.instance_count=4
   ```

4. **Auto-scale compute** for cost savings:
   ```bash
   az ml compute create --type AmlCompute --min-instances 0 --max-instances 10
   ```

5. **Use managed identity** for secure authentication in production.
