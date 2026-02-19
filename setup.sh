#!/bin/bash

# AI Trading Bot - Quick Setup Script
# Run this after cloning the repository

set -e  # Exit on error

echo "=================================="
echo "AI Trading Bot - Setup"
echo "=================================="

# Check Python version
echo ""
echo "[1/5] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Found Python $python_version"

# Check Ollama
echo ""
echo "[2/5] Checking Ollama..."
if command -v ollama &> /dev/null; then
    echo "✓ Ollama is installed"

    # Check if mixtral is pulled
    if ollama list | grep -q "mixtral"; then
        echo "✓ Mixtral model is available"
    else
        echo "⚠  Mixtral model not found"
        echo "   Run: ollama pull mixtral"
    fi
else
    echo "✗ Ollama not installed"
    echo "   Install from: https://ollama.com/install"
    exit 1
fi

# Create virtual environment if it doesn't exist
echo ""
echo "[3/5] Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment exists"
fi

# Activate virtual environment and install dependencies
echo ""
echo "[4/5] Installing Python dependencies..."
source venv/bin/activate
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
echo "✓ Dependencies installed"

# Setup .env file
echo ""
echo "[5/5] Setting up configuration..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✓ Created .env file"
    echo ""
    echo "⚠  ACTION REQUIRED:"
    echo "   Edit .env and add your Alpaca API credentials"
    echo "   Get them from: https://app.alpaca.markets/paper/dashboard/overview"
else
    echo "✓ .env file exists"
fi

# Create logs directory
mkdir -p logs

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Edit .env with your Alpaca credentials"
echo "2. Run: python config.py (test configuration)"
echo "3. Run: python ai_brain.py (test AI connection)"
echo "4. Run: python backtest_runner.py (run backtest)"
echo "5. Run: python live_trader.py (start live trading)"
echo ""
echo "Read README.md for detailed instructions."
echo ""
