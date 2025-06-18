# Workspace-Contained Caching Setup for LLaVA

This document explains how to ensure all caching, model downloads, and data storage happens within the `/workspace` directory, preventing any external system dependencies.

## Overview

The workspace cache setup ensures that:
- All Hugging Face model downloads and caches stay within `/workspace/cache/huggingface/`
- PyTorch model caches stay within `/workspace/cache/torch/`
- Dataset caches stay within `/workspace/cache/datasets/`
- Pip caches stay within `/workspace/cache/pip/`
- All model weights are stored in `/workspace/weights/` or `/workspace/models/`
- Training checkpoints are saved in `/workspace/checkpoints/`
- Data files are located in `/workspace/data/`

## Directory Structure

After setup, your workspace will have the following structure:

```
/workspace/
├── cache/
│   ├── huggingface/
│   │   ├── hub/          # HF Hub cache
│   │   └── transformers/ # Transformers cache
│   ├── torch/            # PyTorch cache
│   ├── datasets/         # HF Datasets cache
│   └── pip/              # Pip cache
├── models/               # Local model storage
├── weights/              # Model weights
├── data/                 # Training/evaluation data
├── checkpoints/          # Training checkpoints
├── .env                  # Environment variables
└── setup_env.sh          # Bash environment setup
```

## Quick Setup

### 1. Run the Setup Script

```bash
cd /workspace
python3 LLaVA/setup_workspace_cache.py
```

This will:
- Create all necessary cache directories
- Set up environment variables
- Create `.env` and `setup_env.sh` files

### 2. Activate Environment

**For Bash:**
```bash
source /workspace/setup_env.sh
```

**For Python scripts:**
```python
import sys
sys.path.insert(0, '/workspace')
import setup_workspace_cache
```

## Environment Variables Set

The following environment variables are configured:

- `WORKSPACE_ROOT=/workspace`
- `HF_HOME=/workspace/cache/huggingface`
- `HUGGINGFACE_HUB_CACHE=/workspace/cache/huggingface/hub`
- `TRANSFORMERS_CACHE=/workspace/cache/huggingface/transformers`
- `TORCH_HOME=/workspace/cache/torch`
- `HF_DATASETS_CACHE=/workspace/cache/datasets`
- `PIP_CACHE_DIR=/workspace/cache/pip`
- `XDG_CACHE_HOME=/workspace/cache`

## Modified Components

### 1. Enhanced Prediction Script (`predict.py`)

The `predict.py` file has been modified to:
- Setup workspace caching at startup
- Ensure all model downloads go to `/workspace/weights/`
- Use workspace-contained cache directories

### 2. Workspace Model Builder (`llava/model/workspace_builder.py`)

A new model builder that:
- Automatically configures cache directories
- Resolves model paths to workspace locations
- Ensures all `from_pretrained` calls use workspace caching

### 3. Workspace Training Script (`llava/train/workspace_train.py`)

An enhanced training script that:
- Sets up workspace environment before any imports
- Ensures training outputs go to `/workspace/checkpoints/`
- Redirects data paths to `/workspace/data/`
- Forces all caching to workspace directories

### 4. Workspace CLIP Encoder (`llava/model/multimodal_encoder/workspace_clip_encoder.py`)

A workspace-aware CLIP encoder that:
- Uses workspace cache for all vision model downloads
- Ensures CLIP model caching stays contained

## Usage Examples

### Using the Workspace Training Script

```bash
# Ensure environment is set up
source /workspace/setup_env.sh

# Run training with workspace caching
python3 /workspace/LLaVA/llava/train/workspace_train.py \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --data_path /workspace/data/llava_instruct.json \
    --image_folder /workspace/data/images \
    --output_dir /workspace/checkpoints/llava-7b-finetune
```

### Using the Workspace Model Builder

```python
from llava.model.workspace_builder import load_pretrained_model_workspace

# This will automatically use workspace caching
tokenizer, model, image_processor, context_len = load_pretrained_model_workspace(
    model_path="liuhaotian/llava-v1.5-7b",
    model_base=None,
    model_name="llava-v1.5-7b"
)
```

### Using the Enhanced Prediction Script

```python
# The predict.py script automatically sets up workspace caching
# Just run it normally and all caching will be contained
python3 predict.py
```

## Manual Cache Directory Setup

If you need to manually set up cache directories in a Python script:

```python
import os

# Set up workspace cache
WORKSPACE_ROOT = "/workspace"
cache_dirs = [
    "cache/huggingface/hub",
    "cache/huggingface/transformers",
    "cache/torch",
    "cache/datasets",
    "cache/pip",
    "models",
    "weights",
    "data",
    "checkpoints"
]

for cache_dir in cache_dirs:
    os.makedirs(os.path.join(WORKSPACE_ROOT, cache_dir), exist_ok=True)

# Set environment variables
os.environ["HF_HOME"] = os.path.join(WORKSPACE_ROOT, "cache/huggingface")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(WORKSPACE_ROOT, "cache/huggingface/hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(WORKSPACE_ROOT, "cache/huggingface/transformers")
os.environ["TORCH_HOME"] = os.path.join(WORKSPACE_ROOT, "cache/torch")
```

## Verification

To verify that caching is working correctly:

1. Check that cache directories exist:
   ```bash
   ls -la /workspace/cache/
   ```

2. Check environment variables:
   ```bash
   env | grep -E "(HF_|TORCH_|TRANSFORMERS_|WORKSPACE_)"
   ```

3. Monitor cache usage during model loading:
   ```bash
   du -sh /workspace/cache/huggingface/hub/
   ```

## Troubleshooting

### Cache Not Being Used

If models are still downloading to system locations:
1. Ensure environment variables are set before importing transformers
2. Check that `setup_workspace_cache.py` was run successfully
3. Verify that `source /workspace/setup_env.sh` was executed

### Permission Issues

If you encounter permission issues:
```bash
chmod -R 755 /workspace/cache/
chmod -R 755 /workspace/models/
chmod -R 755 /workspace/weights/
```

### Disk Space

Monitor disk usage to ensure you have sufficient space:
```bash
df -h /workspace
du -sh /workspace/cache/
```

## Integration with Existing Scripts

To integrate workspace caching with existing scripts, add this at the top:

```python
import os
import sys

# Setup workspace cache before any model imports
WORKSPACE_ROOT = "/workspace"
os.environ["HF_HOME"] = os.path.join(WORKSPACE_ROOT, "cache/huggingface")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(WORKSPACE_ROOT, "cache/huggingface/hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(WORKSPACE_ROOT, "cache/huggingface/transformers")

# Now import your model libraries
from transformers import AutoTokenizer, AutoModelForCausalLM
```

This ensures all subsequent model operations use workspace-contained caching. 