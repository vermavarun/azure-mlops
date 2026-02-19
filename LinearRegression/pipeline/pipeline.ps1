#!/usr/bin/env pwsh
# MLOps Pipeline Helper Script for PowerShell

function Install-Dependencies {
    Write-Host "Installing Python dependencies..." -ForegroundColor Green
    pip install -r requirements.txt
    Write-Host "Dependencies installed successfully!" -ForegroundColor Green
}

function Run-LocalPipeline {
    Write-Host "Running local pipeline..." -ForegroundColor Green
    python app.py
}

function Run-AzurePipeline {
    Write-Host "Running pipeline on Azure ML..." -ForegroundColor Green
    python app.py --azure
}

function Run-Demo {
    Write-Host "Running demo script..." -ForegroundColor Green
    python demo.py
}

function Setup-Environment {
    Write-Host "Setting up environment..." -ForegroundColor Green
    
    # Create necessary directories
    New-Item -ItemType Directory -Path "data", "models", "metrics", "outputs", "logs" -Force | Out-Null
    
    # Copy example .env if it doesn't exist
    if (-not (Test-Path ".env")) {
        Copy-Item ".env.example" ".env" -Force
        Write-Host ".env file created from .env.example" -ForegroundColor Yellow
        Write-Host "Please update .env with your Azure credentials" -ForegroundColor Yellow
    }
    
    Write-Host "Environment setup completed!" -ForegroundColor Green
}

function Show-Help {
    Write-Host "`nMLOps Pipeline Helper Script`n" -ForegroundColor Cyan
    Write-Host "Usage: .\pipeline.ps1 [command]`n" -ForegroundColor White
    Write-Host "Commands:" -ForegroundColor Green
    Write-Host "  setup     - Initialize environment and directories"
    Write-Host "  install   - Install Python dependencies"
    Write-Host "  local     - Run pipeline locally"
    Write-Host "  azure     - Run pipeline on Azure ML"
    Write-Host "  demo      - Run demonstration examples"
    Write-Host "  help      - Show this help message`n"
}

# Main script logic
$command = $args[0].ToLower()

switch ($command) {
    "setup" { Setup-Environment }
    "install" { Install-Dependencies }
    "local" { Run-LocalPipeline }
    "azure" { Run-AzurePipeline }
    "demo" { Run-Demo }
    "help" { Show-Help }
    "" { Show-Help }
    default {
        Write-Host "Unknown command: $command" -ForegroundColor Red
        Show-Help
    }
}
