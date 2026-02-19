# Azure MLOps Infrastructure for Linear Regression

This directory contains Terraform configurations for deploying Azure infrastructure to support an MLOps pipeline for linear regression model training and deployment.

## Architecture Overview

The infrastructure includes:

- **Azure Resource Group**: Container for all resources
- **Azure Storage Account**: For datasets and model storage
- **Azure Container Registry**: For Docker image management
- **Azure Key Vault**: For secrets and credentials management
- **Azure ML Workspace**: Main MLOps platform
- **Application Insights**: For monitoring and logging
- **Virtual Network**: Networking and security
- **Compute Resources**: Compute instance and compute cluster for training

## Files

- `providers.tf`: Terraform provider configuration
- `variables.tf`: Variable definitions with defaults
- `main.tf`: Main infrastructure resources
- `outputs.tf`: Output definitions
- `terraform.tfvars.example`: Example variables file

## Prerequisites

1. Azure CLI installed and authenticated
2. Terraform >= 1.0 installed
3. Azure subscription with appropriate permissions

## Setup Instructions

### 1. Authenticate with Azure

```bash
az login
az account set --subscription "<your-subscription-id>"
```

### 2. Prepare Terraform Variables

```bash
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your desired values
```

### 3. Initialize Terraform

```bash
terraform init
```

### 4. Plan Deployment

```bash
terraform plan -out=tfplan
```

### 5. Apply Configuration

```bash
terraform apply tfplan
```

## Configuration Variables

Key variables you can customize:

- `project_name`: Name prefix for all resources (default: mlops-lr)
- `environment`: Environment name (dev, staging, prod)
- `location`: Azure region (default: eastus)
- `ml_workspace_sku`: ML workspace SKU (basic or standard)
- `compute_instance_vm_size`: VM size for interactive development
- `cluster_min_nodes`: Minimum nodes in compute cluster
- `cluster_max_nodes`: Maximum nodes in compute cluster

## Outputs

After successful deployment, the following outputs are available:

- `aml_workspace_name`: Name of the Azure ML workspace
- `storage_account_name`: Name of the storage account
- `container_registry_login_server`: ACR login server URL
- `key_vault_name`: Name of the Key Vault
- And more... (see outputs.tf)

To view outputs:

```bash
terraform output
```

## Cleanup

To destroy all resources:

```bash
terraform destroy
```

## Next Steps

After infrastructure deployment:

1. Connect to the compute instance to begin development
2. Upload training data to the storage account
3. Set up CI/CD pipelines for model training
4. Configure monitoring and alerts

## Cost Considerations

- The `basic` ML workspace SKU has limited features
- Consider `Standard` SKU for production
- Compute instances incur costs while running
- Enable autoscaling on compute clusters to manage costs
- Use low-priority VMs for non-critical workloads

## Security Notes

- The Key Vault enables soft delete and purge protection
- Storage containers are set to private access
- Network Security Group is configured for the compute subnet
- Managed identities are used for secure service-to-service communication
