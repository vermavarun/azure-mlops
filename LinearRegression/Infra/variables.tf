variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "mlops-lr"
}

variable "environment" {
  description = "Environment name (dev, prod, staging)"
  type        = string
  default     = "dev"
}

variable "location" {
  description = "Azure region for resources"
  type        = string
  default     = "eastus"
}

variable "tags" {
  description = "Common tags for all resources"
  type        = map(string)
  default = {
    Project     = "MLOps Linear Regression"
    Environment = "dev"
    ManagedBy   = "Terraform"
  }
}

variable "storage_account_tier" {
  description = "Storage account tier"
  type        = string
  default     = "Standard"
}

variable "storage_account_replication_type" {
  description = "Storage account replication type"
  type        = string
  default     = "LRS"
}

variable "key_vault_sku" {
  description = "Key Vault SKU"
  type        = string
  default     = "standard"
}

variable "compute_instance_vm_size" {
  description = "VM size for compute instance"
  type        = string
  default     = "Standard_DS2_v2"
}

variable "enable_aml_cluster" {
  description = "Enable Azure ML compute cluster"
  type        = bool
  default     = true
}

variable "cluster_min_nodes" {
  description = "Minimum nodes in ML cluster"
  type        = number
  default     = 0
}

variable "cluster_max_nodes" {
  description = "Maximum nodes in ML cluster"
  type        = number
  default     = 2
}
