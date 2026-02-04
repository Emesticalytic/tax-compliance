#!/bin/bash

# Tax Compliance Analysis - Quick Start Script
# This script automates the complete setup process

set -e  # Exit on error

echo "======================================================================"
echo "🔍 Tax Compliance Risk Analysis - Quick Start"
echo "====================================================================="
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python --version 2>&1)
echo "   Found: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "📦 Creating virtual environment..."
    python -m venv venv
    echo "   ✓ Virtual environment created"
else
    echo ""
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "🔌 Activating virtual environment..."
source venv/bin/activate
echo "   ✓ Virtual environment activated"

# Install dependencies
echo ""
echo "📚 Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "   ✓ All dependencies installed"

# Run the pipeline
echo ""
echo "======================================================================"
echo "🚀 Running ML Pipeline"
echo "======================================================================"
echo ""
python main.py

# Check outputs
echo ""
echo "======================================================================"
echo "✅ Setup Complete!"
echo "======================================================================"
echo ""
echo "📁 Generated Files:"
echo "   • data/raw/synthetic_taxpayers.csv - 10K taxpayer records"
echo "   • output/model/risk_model.pkl - Trained model"
echo "   • output/eda/*.png - 5 EDA visualizations"
echo "   • output/model/*.png - 5 evaluation charts"
echo ""
echo "======================================================================"
echo "🌐 Next Steps"
echo "======================================================================"
echo ""
echo "1. Launch Streamlit Dashboard:"
echo "   $ streamlit run streamlit_app.py"
echo ""
echo "2. Run Tests:"
echo "   $ pytest tests/ -v"
echo ""
echo "3. Push to GitHub:"
echo "   $ git init"
echo "   $ git add ."
echo "   $ git commit -m 'Initial commit: Tax compliance analysis'"
echo "   $ git remote add origin <your-repo-url>"
echo "   $ git push -u origin main"
echo ""
echo "======================================================================"
echo "📖 Documentation"
echo "======================================================================"
echo ""
echo "• README.md - Full project documentation"
echo "• SETUP_GUIDE.md - Detailed setup instructions"
echo "• INTERVIEW_GUIDE.md - Interview preparation tips"
echo ""
echo "======================================================================"
echo "Good luck with your interview! 🎉"
echo "======================================================================"
