@echo off
REM Quick Start Script for Qwen 14B Fine-tuning
REM This script sets up the environment and starts training

echo ========================================
echo Qwen 14B Fine-tuning Quick Start
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10 or higher
    pause
    exit /b 1
)

echo [1/4] Checking Python version...
python --version

REM Check if CUDA is available
echo.
echo [2/4] Checking CUDA availability...
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>nul
if errorlevel 1 (
    echo WARNING: PyTorch not installed yet. Will install dependencies...
)

REM Check if requirements are installed
echo.
echo [3/4] Checking dependencies...
pip show unsloth >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies... This may take a while...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
) else (
    echo Dependencies already installed!
)

REM Check if dataset exists
echo.
echo [4/4] Checking dataset...
if not exist "dataset.json" (
    echo ERROR: dataset.json not found!
    echo Please ensure dataset.json is in the same directory
    pause
    exit /b 1
)
echo Dataset found: dataset.json

REM Start training
echo.
echo ========================================
echo Starting Fine-tuning Process
echo ========================================
echo.
echo This will take several hours depending on your GPU.
echo You can monitor progress in the console output.
echo.
echo Press Ctrl+C to stop training at any time.
echo.
pause

python finetune_qwen14b_unsloth.py

if errorlevel 1 (
    echo.
    echo ========================================
    echo Training failed with errors!
    echo ========================================
    pause
    exit /b 1
)

echo.
echo ========================================
echo Training Completed Successfully!
echo ========================================
echo.
echo Your fine-tuned model is ready in: qwen14b_finetuned/
echo.
echo Next steps:
echo 1. Check the output directory for your models
echo 2. Use the GGUF files with Ollama (see README.md)
echo 3. Test inference with the provided examples
echo.
pause
