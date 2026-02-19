# Create Resource Group
resource "azurerm_resource_group" "rg" {
  name     = "${var.project_name}-${var.environment}-rg-${local.unique_suffix}"
  location = var.location
  tags     = var.tags
}

# Create Storage Account for datasets and models
resource "azurerm_storage_account" "storage" {
  name                     = replace("${var.project_name}${var.environment}sa${local.unique_suffix}", "-", "")
  resource_group_name      = azurerm_resource_group.rg.name
  location                 = azurerm_resource_group.rg.location
  account_tier             = var.storage_account_tier
  account_replication_type = var.storage_account_replication_type
  tags                     = var.tags
}

# Create Storage Container for training data
resource "azurerm_storage_container" "training_data" {
  name                  = "training-data"
  storage_account_name  = azurerm_storage_account.storage.name
  container_access_type = "private"
}

# Create Storage Container for models
resource "azurerm_storage_container" "models" {
  name                  = "models"
  storage_account_name  = azurerm_storage_account.storage.name
  container_access_type = "private"
}

# Create Container Registry for Docker images
resource "azurerm_container_registry" "acr" {
  name                = replace("${var.project_name}${var.environment}acr${local.unique_suffix}", "-", "")
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  sku                 = "Basic"
  admin_enabled       = true
  tags                = var.tags
}

# Create Key Vault
resource "azurerm_key_vault" "kv" {
  name                        = "${var.project_name}-${var.environment}-kv-${substr(uuid(), 0, 8)}"
  location                    = azurerm_resource_group.rg.location
  resource_group_name         = azurerm_resource_group.rg.name
  enabled_for_disk_encryption = true
  tenant_id                   = data.azurerm_client_config.current.tenant_id
  sku_name                    = var.key_vault_sku
  purge_protection_enabled    = false
  soft_delete_retention_days  = 7
  tags                        = var.tags
}

# Key Vault Access Policy for current user/deployer
resource "azurerm_key_vault_access_policy" "deployer_access" {
  key_vault_id       = azurerm_key_vault.kv.id
  tenant_id          = data.azurerm_client_config.current.tenant_id
  object_id          = data.azurerm_client_config.current.object_id

  secret_permissions = [
    "Get",
    "List",
    "Set",
    "Delete",
    "Recover",
    "Purge"
  ]
}


# Application Insights for monitoring
resource "azurerm_application_insights" "appinsights" {
  name                       = "${var.project_name}-${var.environment}-appinsights"
  location                   = azurerm_resource_group.rg.location
  resource_group_name        = azurerm_resource_group.rg.name
  application_type           = "web"
  retention_in_days          = 30
  tags                       = var.tags
}

# Azure ML Workspace
resource "azurerm_machine_learning_workspace" "aml_workspace" {
  name                    = "${var.project_name}-${var.environment}-amlws-${substr(uuid(), 0, 8)}"
  location                = azurerm_resource_group.rg.location
  resource_group_name     = azurerm_resource_group.rg.name
  application_insights_id = azurerm_application_insights.appinsights.id
  key_vault_id            = azurerm_key_vault.kv.id
  storage_account_id      = azurerm_storage_account.storage.id
  public_network_access_enabled = true
  tags                    = var.tags

  identity {
    type = "SystemAssigned"
  }
}

# Virtual Network
resource "azurerm_virtual_network" "vnet" {
  name                = "${var.project_name}-${var.environment}-vnet"
  address_space       = ["10.0.0.0/16"]
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  tags                = var.tags
}

# Subnet for compute resources
resource "azurerm_subnet" "compute_subnet" {
  name                 = "compute-subnet"
  resource_group_name  = azurerm_resource_group.rg.name
  virtual_network_name = azurerm_virtual_network.vnet.name
  address_prefixes     = ["10.0.1.0/24"]
}

# Network Security Group
resource "azurerm_network_security_group" "nsg" {
  name                = "${var.project_name}-${var.environment}-nsg"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  tags                = var.tags
}

# Associate NSG with subnet
resource "azurerm_subnet_network_security_group_association" "nsg_association" {
  subnet_id                 = azurerm_subnet.compute_subnet.id
  network_security_group_id = azurerm_network_security_group.nsg.id
}

# Get current user context
data "azurerm_client_config" "current" {}

# Generate unique suffix
locals {
  unique_suffix = substr(uuid(), 0, 8)
}

# Note: Azure ML Workspace automatically creates role assignments for storage and key vault
# ACR role assignment for the workspace
resource "azurerm_role_assignment" "acr_access" {
  scope              = azurerm_container_registry.acr.id
  role_definition_name = "AcrPull"
  principal_id       = azurerm_machine_learning_workspace.aml_workspace.identity[0].principal_id
}

# Compute Instance for interactive development
resource "azurerm_machine_learning_compute_instance" "compute_instance" {
  name                          = "${var.project_name}-${var.environment}-ci"
  machine_learning_workspace_id = azurerm_machine_learning_workspace.aml_workspace.id
  virtual_machine_size          = var.compute_instance_vm_size
  subnet_resource_id            = azurerm_subnet.compute_subnet.id
  tags                          = var.tags
}

# Compute Cluster for training jobs
resource "azurerm_machine_learning_compute_cluster" "compute_cluster" {
  count                         = var.enable_aml_cluster ? 1 : 0
  name                          = "${var.project_name}-${var.environment}-cc"
  location                      = azurerm_resource_group.rg.location
  machine_learning_workspace_id = azurerm_machine_learning_workspace.aml_workspace.id
  vm_priority                   = "Dedicated"
  vm_size                       = "Standard_DS2_v2"
  subnet_resource_id            = azurerm_subnet.compute_subnet.id

  scale_settings {
    min_node_count                       = var.cluster_min_nodes
    max_node_count                       = var.cluster_max_nodes
    scale_down_nodes_after_idle_duration = "PT5M"
  }

  tags = var.tags
}
