#!/bin/bash

# Setup script for Precision Indoor Cultivation Project

echo "🌱 Setting up Precision Indoor Cultivation Project..."
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing required packages..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the dashboard:"
echo "  streamlit run app.py"
echo ""
echo "To train models:"
echo "  python pH.py"
echo "  python NHI.py"
echo "  python plantgrowthstageclassification.py"

