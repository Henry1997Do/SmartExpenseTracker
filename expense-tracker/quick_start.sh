#!/bin/bash

###############################################################################
# Smart Expense Tracker - Quick Start Script
# Automated setup for the expense tracking application
###############################################################################

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Print banner
print_banner() {
    echo ""
    echo -e "${PURPLE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${PURPLE}║                                                            ║${NC}"
    echo -e "${PURPLE}║         💰 SMART EXPENSE TRACKER - QUICK START 💰          ║${NC}"
    echo -e "${PURPLE}║                                                            ║${NC}"
    echo -e "${PURPLE}║              AI-Powered Financial Management               ║${NC}"
    echo -e "${PURPLE}║                                                            ║${NC}"
    echo -e "${PURPLE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# Print step header
print_step() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# Print success message
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# Print error message
print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Print warning message
print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Print info message
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Main execution
main() {
    print_banner
    
    # Step 1: Check Python installation
    print_step "STEP 1: Checking Python Installation"
    
    if command_exists python3; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        print_success "Python 3 is installed (Version: $PYTHON_VERSION)"
        PYTHON_CMD="python3"
    elif command_exists python; then
        PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
        if [[ $PYTHON_VERSION == 3.* ]]; then
            print_success "Python 3 is installed (Version: $PYTHON_VERSION)"
            PYTHON_CMD="python"
        else
            print_error "Python 3 is required but Python $PYTHON_VERSION is installed"
            print_info "Please install Python 3.8 or higher from https://www.python.org/"
            exit 1
        fi
    else
        print_error "Python is not installed"
        print_info "Please install Python 3.8 or higher from https://www.python.org/"
        exit 1
    fi
    
    # Check Python version is 3.8+
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
        print_error "Python 3.8 or higher is required (found $PYTHON_VERSION)"
        print_info "Please upgrade Python from https://www.python.org/"
        exit 1
    fi
    
    # Step 2: Check pip installation
    print_step "STEP 2: Checking pip Installation"
    
    if command_exists pip3; then
        print_success "pip3 is installed"
        PIP_CMD="pip3"
    elif command_exists pip; then
        print_success "pip is installed"
        PIP_CMD="pip"
    else
        print_error "pip is not installed"
        print_info "Please install pip: https://pip.pypa.io/en/stable/installation/"
        exit 1
    fi
    
    # Step 3: Install dependencies
    print_step "STEP 3: Installing Dependencies"
    
    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt not found!"
        print_info "Please ensure you're in the correct directory"
        exit 1
    fi
    
    print_info "Installing Python packages (this may take 2-3 minutes)..."
    $PIP_CMD install -r requirements.txt --quiet
    
    if [ $? -eq 0 ]; then
        print_success "All dependencies installed successfully"
    else
        print_error "Failed to install dependencies"
        print_info "Try running: $PIP_CMD install -r requirements.txt"
        exit 1
    fi
    
    # Step 4: Generate training data
    print_step "STEP 4: Generating Training Data"
    
    if [ ! -f "generate_data.py" ]; then
        print_error "generate_data.py not found!"
        exit 1
    fi
    
    print_info "Creating 1,500 synthetic transactions..."
    $PYTHON_CMD generate_data.py
    
    if [ $? -eq 0 ] && [ -f "expenses.csv" ]; then
        print_success "Training data generated successfully"
        TRANSACTION_COUNT=$(wc -l < expenses.csv)
        print_info "Created $TRANSACTION_COUNT transactions in expenses.csv"
    else
        print_error "Failed to generate training data"
        exit 1
    fi
    
    # Step 5: Train ML model
    print_step "STEP 5: Training Machine Learning Model"
    
    if [ ! -f "train_model.py" ]; then
        print_error "train_model.py not found!"
        exit 1
    fi
    
    print_info "Training AI models (this may take 1-2 minutes)..."
    $PYTHON_CMD train_model.py
    
    if [ $? -eq 0 ] && [ -f "expense_model.pkl" ]; then
        print_success "ML model trained successfully"
        
        # Check for all required files
        if [ -f "vectorizer.pkl" ] && [ -f "label_encoder.pkl" ]; then
            print_success "All model files created"
        fi
        
        if [ -f "confusion_matrix.png" ] && [ -f "category_accuracy.png" ]; then
            print_success "Evaluation charts generated"
        fi
    else
        print_error "Failed to train ML model"
        exit 1
    fi
    
    # Step 6: Verify setup
    print_step "STEP 6: Verifying Setup"
    
    SETUP_COMPLETE=true
    
    # Check all required files
    FILES=("expenses.csv" "expense_model.pkl" "vectorizer.pkl" "label_encoder.pkl" "app.py")
    
    for file in "${FILES[@]}"; do
        if [ -f "$file" ]; then
            print_success "$file ✓"
        else
            print_error "$file not found"
            SETUP_COMPLETE=false
        fi
    done
    
    # Final message
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    if [ "$SETUP_COMPLETE" = true ]; then
        echo ""
        echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║                                                            ║${NC}"
        echo -e "${GREEN}║                  🎉 SETUP COMPLETE! 🎉                     ║${NC}"
        echo -e "${GREEN}║                                                            ║${NC}"
        echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo -e "${YELLOW}📊 Setup Summary:${NC}"
        echo -e "   ${GREEN}✅${NC} Python $PYTHON_VERSION installed"
        echo -e "   ${GREEN}✅${NC} All dependencies installed"
        echo -e "   ${GREEN}✅${NC} Training data generated (1,500 transactions)"
        echo -e "   ${GREEN}✅${NC} ML model trained and saved"
        echo -e "   ${GREEN}✅${NC} All files verified"
        echo ""
        echo -e "${CYAN}🚀 Next Steps:${NC}"
        echo ""
        echo -e "   ${YELLOW}1.${NC} Start the application:"
        echo -e "      ${GREEN}streamlit run app.py${NC}"
        echo ""
        echo -e "   ${YELLOW}2.${NC} Open your browser to:"
        echo -e "      ${BLUE}http://localhost:8501${NC}"
        echo ""
        echo -e "   ${YELLOW}3.${NC} Start tracking your expenses!"
        echo ""
        echo -e "${PURPLE}💡 Tips:${NC}"
        echo -e "   • Read ${CYAN}README.md${NC} for detailed documentation"
        echo -e "   • Check ${CYAN}DAY_TIMELINE.md${NC} for a structured guide"
        echo -e "   • Try the AI categorization feature!"
        echo ""
        echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""
        echo -e "${GREEN}Happy Tracking! 💰📊✨${NC}"
        echo ""
    else
        echo ""
        echo -e "${RED}╔════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${RED}║                                                            ║${NC}"
        echo -e "${RED}║                  ⚠️  SETUP INCOMPLETE ⚠️                   ║${NC}"
        echo -e "${RED}║                                                            ║${NC}"
        echo -e "${RED}╚════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo -e "${YELLOW}Some files are missing. Please check the errors above.${NC}"
        echo ""
        echo -e "${CYAN}Troubleshooting:${NC}"
        echo -e "   1. Ensure all project files are in the current directory"
        echo -e "   2. Check for error messages above"
        echo -e "   3. Try running the setup steps manually"
        echo -e "   4. See README.md for detailed instructions"
        echo ""
        exit 1
    fi
}

# Run main function
main
