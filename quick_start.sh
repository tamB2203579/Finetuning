#!/bin/bash
# Quick Start Script for Qwen 14B Fine-tuning (Linux/Mac)
# This script sets up the environment and starts training

echo "========================================"
echo "Qwen 14B Fine-tuning Quick Start"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python is not installed or not in PATH"
    echo "Please install Python 3.10 or higher"
    exit 1
fi

echo "[1/4] Checking Python version..."
python3 --version

# Check if CUDA is available
echo ""
echo "[2/4] Checking CUDA availability..."
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "WARNING: PyTorch not installed yet. Will install dependencies..."

# Check if requirements are installed
echo ""
echo "[3/4] Checking dependencies..."
if ! pip show unsloth &> /dev/null; then
    echo "Installing dependencies... This may take a while..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install dependencies"
        exit 1
    fi
else
    echo "Dependencies already installed!"
fi

# Check if dataset exists
echo ""
echo "[4/4] Checking dataset..."
if [ ! -f "dataset.json" ]; then
    echo "ERROR: dataset.json not found!"
    echo "Please ensure dataset.json is in the same directory"
    exit 1
fi
echo "Dataset found: dataset.json"

# Start training
echo ""
echo "========================================"
echo "Starting Fine-tuning Process"
echo "========================================"
echo ""
echo "This will take several hours depending on your GPU."
echo "You can monitor progress in the console output."
echo ""
echo "Press Ctrl+C to stop training at any time."
echo ""
read -p "Press Enter to continue..."

python3 finetune_qwen14b_unsloth.py

if [ $? -ne 0 ]; then
    echo ""
    echo "========================================"
    echo "Training failed with errors!"
    echo "========================================"
    exit 1
fi

echo ""
echo "========================================"
echo "Training Completed Successfully!"
echo "========================================"
echo ""
echo "Your fine-tuned model is ready in: qwen14b_finetuned/"
echo ""
echo "Next steps:"
echo "1. Check the output directory for your models"
echo "2. Use the GGUF files with Ollama (see README.md)"
echo "3. Test inference with the provided examples"
echo ""
