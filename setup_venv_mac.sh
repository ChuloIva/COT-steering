#!/bin/bash

# Setup script for COT-steering project on Mac using venv

set -e  # Exit on any error

echo "🚀 Setting up COT-steering project with Python venv on Mac..."

# Check if Python 3.10+ is available
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Found Python version: $python_version"

if [[ $(echo "$python_version >= 3.10" | bc -l) -eq 0 ]]; then
    echo "❌ Python 3.10+ is required. Please install Python 3.10 or higher."
    echo "You can install it using Homebrew: brew install python@3.10"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv stllms_venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source stllms_venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch first (Mac-specific)
echo "🔥 Installing PyTorch for Mac..."
if [[ $(uname -m) == "arm64" ]]; then
    # Apple Silicon Mac
    echo "Detected Apple Silicon Mac - installing PyTorch with MPS support"
    pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1
else
    # Intel Mac
    echo "Detected Intel Mac - installing PyTorch for CPU"
    pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
echo "📚 Installing other dependencies..."
pip install -r requirements_mac.txt

# Install the package in development mode
echo "🔧 Installing package in development mode..."
pip install -e .

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source stllms_venv/bin/activate"
echo ""
echo "To deactivate the environment, run:"
echo "  deactivate"
echo ""
echo "🎯 You can now run the project commands!"
