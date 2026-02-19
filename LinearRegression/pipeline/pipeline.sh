#!/bin/bash
# MLOps Pipeline Helper Script for Bash/Linux/Mac

set -e

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

install_dependencies() {
    echo -e "${GREEN}Installing Python dependencies...${NC}"
    pip install -r requirements.txt
    echo -e "${GREEN}Dependencies installed successfully!${NC}"
}

run_local_pipeline() {
    echo -e "${GREEN}Running local pipeline...${NC}"
    python app.py
}

run_azure_pipeline() {
    echo -e "${GREEN}Running pipeline on Azure ML...${NC}"
    python app.py --azure
}

run_demo() {
    echo -e "${GREEN}Running demo script...${NC}"
    python demo.py
}

setup_environment() {
    echo -e "${GREEN}Setting up environment...${NC}"
    
    # Create necessary directories
    mkdir -p data models metrics outputs logs
    
    # Copy example .env if it doesn't exist
    if [ ! -f ".env" ]; then
        cp .env.example .env
        echo -e "${YELLOW}.env file created from .env.example${NC}"
        echo -e "${YELLOW}Please update .env with your Azure credentials${NC}"
    fi
    
    echo -e "${GREEN}Environment setup completed!${NC}"
}

show_help() {
    cat << EOF

${CYAN}MLOps Pipeline Helper Script${NC}

${CYAN}Usage:${NC} ./pipeline.sh [command]

${CYAN}Commands:${NC}
  setup     - Initialize environment and directories
  install   - Install Python dependencies
  local     - Run pipeline locally
  azure     - Run pipeline on Azure ML
  demo      - Run demonstration examples
  help      - Show this help message

EOF
}

# Main script logic
command=$1

case "$command" in
    setup)
        setup_environment
        ;;
    install)
        install_dependencies
        ;;
    local)
        run_local_pipeline
        ;;
    azure)
        run_azure_pipeline
        ;;
    demo)
        run_demo
        ;;
    help)
        show_help
        ;;
    *)
        if [ -z "$command" ]; then
            show_help
        else
            echo -e "${RED}Unknown command: $command${NC}"
            show_help
        fi
        ;;
esac
