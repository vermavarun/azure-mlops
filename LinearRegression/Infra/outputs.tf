output "resource_group_name" {
  description = "Name of the resource group"
  value       = azurerm_resource_group.rg.name
}

output "resource_group_id" {
  description = "ID of the resource group"
  value       = azurerm_resource_group.rg.id
}

output "storage_account_name" {
  description = "Name of the storage account"
  value       = azurerm_storage_account.storage.name
}

output "storage_account_id" {
  description = "ID of the storage account"
  value       = azurerm_storage_account.storage.id
}

output "container_registry_name" {
  description = "Name of the container registry"
  value       = azurerm_container_registry.acr.name
}

output "container_registry_login_server" {
  description = "Login server of the container registry"
  value       = azurerm_container_registry.acr.login_server
}

output "key_vault_name" {
  description = "Name of the key vault"
  value       = azurerm_key_vault.kv.name
}

output "key_vault_id" {
  description = "ID of the key vault"
  value       = azurerm_key_vault.kv.id
}

output "aml_workspace_name" {
  description = "Name of the Azure ML workspace"
  value       = azurerm_machine_learning_workspace.aml_workspace.name
}

output "aml_workspace_id" {
  description = "ID of the Azure ML workspace"
  value       = azurerm_machine_learning_workspace.aml_workspace.id
}

output "application_insights_name" {
  description = "Name of the Application Insights instance"
  value       = azurerm_application_insights.appinsights.name
}

output "application_insights_instrumentation_key" {
  description = "Instrumentation key for Application Insights"
  value       = azurerm_application_insights.appinsights.instrumentation_key
  sensitive   = true
}

output "vnet_id" {
  description = "ID of the virtual network"
  value       = azurerm_virtual_network.vnet.id
}

output "vnet_name" {
  description = "Name of the virtual network"
  value       = azurerm_virtual_network.vnet.name
}

output "compute_instance_name" {
  description = "Name of the compute instance"
  value       = azurerm_machine_learning_compute_instance.compute_instance.name
}

output "compute_cluster_name" {
  description = "Name of the compute cluster"
  value       = try(azurerm_machine_learning_compute_cluster.compute_cluster[0].name, "Not created")
}

output "training_data_container" {
  description = "Training data storage container name"
  value       = azurerm_storage_container.training_data.name
}

output "models_container" {
  description = "Models storage container name"
  value       = azurerm_storage_container.models.name
}
