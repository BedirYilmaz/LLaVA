#!/usr/bin/env python3
"""
Setup script to ensure all cache and model directories are contained within /workspace
"""
import os
import sys

# Get the absolute path to the workspace
WORKSPACE_ROOT = "/workspace"
if not os.path.exists(WORKSPACE_ROOT):
    WORKSPACE_ROOT = os.path.abspath(os.path.dirname(__file__))

# Create necessary cache directories
CACHE_DIRS = [
    "cache/huggingface/hub",
    "cache/huggingface/transformers", 
    "cache/torch",
    "cache/datasets",
    "cache/pip",
    "models",
    "data",
    "weights"
]

def setup_cache_directories():
    """Create all necessary cache directories within workspace"""
    print(f"Setting up cache directories in {WORKSPACE_ROOT}")
    
    for cache_dir in CACHE_DIRS:
        full_path = os.path.join(WORKSPACE_ROOT, cache_dir)
        os.makedirs(full_path, exist_ok=True)
        print(f"Created/ensured directory: {full_path}")

def setup_environment_variables():
    """Set environment variables to point to workspace cache directories"""
    env_vars = {
        # Hugging Face cache locations
        "HF_HOME": os.path.join(WORKSPACE_ROOT, "cache/huggingface"),
        "HUGGINGFACE_HUB_CACHE": os.path.join(WORKSPACE_ROOT, "cache/huggingface/hub"),
        "TRANSFORMERS_CACHE": os.path.join(WORKSPACE_ROOT, "cache/huggingface/transformers"),
        
        # PyTorch cache
        "TORCH_HOME": os.path.join(WORKSPACE_ROOT, "cache/torch"),
        
        # Datasets cache
        "HF_DATASETS_CACHE": os.path.join(WORKSPACE_ROOT, "cache/datasets"),
        
        # Pip cache
        "PIP_CACHE_DIR": os.path.join(WORKSPACE_ROOT, "cache/pip"),
        
        # Additional common cache locations
        "XDG_CACHE_HOME": os.path.join(WORKSPACE_ROOT, "cache"),
        
        # Prevent automatic model downloads to system locations
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        
        # Force workspace as working directory
        "WORKSPACE_ROOT": WORKSPACE_ROOT
    }
    
    print("Setting environment variables:")
    for var, value in env_vars.items():
        os.environ[var] = value
        print(f"  {var}={value}")
    
    return env_vars

def create_env_file():
    """Create a .env file with all the environment variables"""
    env_vars = {
        "HF_HOME": os.path.join(WORKSPACE_ROOT, "cache/huggingface"),
        "HUGGINGFACE_HUB_CACHE": os.path.join(WORKSPACE_ROOT, "cache/huggingface/hub"),
        "TRANSFORMERS_CACHE": os.path.join(WORKSPACE_ROOT, "cache/huggingface/transformers"),
        "TORCH_HOME": os.path.join(WORKSPACE_ROOT, "cache/torch"),
        "HF_DATASETS_CACHE": os.path.join(WORKSPACE_ROOT, "cache/datasets"),
        "PIP_CACHE_DIR": os.path.join(WORKSPACE_ROOT, "cache/pip"),
        "XDG_CACHE_HOME": os.path.join(WORKSPACE_ROOT, "cache"),
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "WORKSPACE_ROOT": WORKSPACE_ROOT
    }
    
    env_file_path = os.path.join(WORKSPACE_ROOT, ".env")
    with open(env_file_path, "w") as f:
        f.write("# Environment variables to keep all cache within workspace\n")
        for var, value in env_vars.items():
            f.write(f"{var}={value}\n")
    
    print(f"Created .env file at {env_file_path}")

def create_bashrc_additions():
    """Create a script to source for bash environments"""
    bashrc_path = os.path.join(WORKSPACE_ROOT, "setup_env.sh")
    
    with open(bashrc_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Source this file to set up workspace cache environment\n")
        f.write("# Usage: source setup_env.sh\n\n")
        
        f.write(f'export WORKSPACE_ROOT="{WORKSPACE_ROOT}"\n')
        f.write(f'export HF_HOME="$WORKSPACE_ROOT/cache/huggingface"\n')
        f.write(f'export HUGGINGFACE_HUB_CACHE="$WORKSPACE_ROOT/cache/huggingface/hub"\n')
        f.write(f'export TRANSFORMERS_CACHE="$WORKSPACE_ROOT/cache/huggingface/transformers"\n')
        f.write(f'export TORCH_HOME="$WORKSPACE_ROOT/cache/torch"\n')
        f.write(f'export HF_DATASETS_CACHE="$WORKSPACE_ROOT/cache/datasets"\n')
        f.write(f'export PIP_CACHE_DIR="$WORKSPACE_ROOT/cache/pip"\n')
        f.write(f'export XDG_CACHE_HOME="$WORKSPACE_ROOT/cache"\n')
        f.write(f'export HF_HUB_DISABLE_SYMLINKS_WARNING="1"\n')
        f.write(f'export HF_HUB_DISABLE_PROGRESS_BARS="1"\n')
        f.write('\necho "Environment configured for workspace-contained caching"\n')
    
    # Make it executable
    os.chmod(bashrc_path, 0o755)
    print(f"Created bash setup script at {bashrc_path}")

if __name__ == "__main__":
    print("Setting up workspace-contained caching...")
    setup_cache_directories()
    setup_environment_variables()
    create_env_file()
    create_bashrc_additions()
    print("\nSetup complete!")
    print(f"All cache directories are now configured within: {WORKSPACE_ROOT}")
    print("\nTo activate in bash:")
    print(f"  source {WORKSPACE_ROOT}/setup_env.sh")
    print("\nTo activate in Python:")
    print(f"  import sys; sys.path.insert(0, '{WORKSPACE_ROOT}'); import setup_workspace_cache") 