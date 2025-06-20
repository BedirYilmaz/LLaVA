#!/bin/bash

# ORPO Training Launcher Script for LLaVA v1.6 7B
# This script activates the virtual environment and runs the ORPO training

echo "🚀 Starting ORPO training for LLaVA v1.6 7B on RLAIF-V dataset"
echo "=================================================="

# Activate virtual environment
source venv/bin/activate

# Check if running with accelerate (recommended for multi-GPU)
if command -v accelerate &> /dev/null; then
    echo "Using accelerate for training..."
    accelerate launch orpo_llava_training.py
else
    echo "Running without accelerate (single GPU)..."
    python orpo_llava_training.py
fi

echo "=================================================="
echo "✅ Training completed!" 