# ORPO Training for LLaVA v1.6 7B

This directory contains scripts and configurations for training LLaVA v1.6 7B using **ORPO (Odds Ratio Preference Optimization)** on the **RLAIF-V dataset**. ORPO is a reference-free preference optimization method that doesn't require a separate reference model, making it more memory-efficient than DPO.

## Overview

- **Model**: LLaVA v1.6 7B (Mistral-based)
- **Method**: ORPO (Odds Ratio Preference Optimization)
- **Dataset**: RLAIF-V (83,000+ preference pairs for vision-language alignment)
- **Optimization**: LoRA (Low-Rank Adaptation) for efficient training
- **Framework**: TRL (Transformers Reinforcement Learning)

## Files Description

- `orpo_llava_training.py` - Main training script for full dataset
- `orpo_llava_training_test.py` - Test script with small subset (100 samples)
- `test_orpo_setup.py` - Validation script to check setup
- `run_orpo_training.sh` - Launcher script with virtual environment activation
- `ORPO_TRAINING_README.md` - This documentation

## Prerequisites

1. **Virtual Environment**: Ensure the `venv` is activated
2. **TRL Library**: Already installed via the setup
3. **RLAIF-V Dataset**: Available in workspace cache
4. **GPU Memory**: Recommended 24GB+ VRAM for full training

## Quick Start

### 1. Test the Setup

First, validate that everything is working correctly:

```bash
source venv/bin/activate
python test_orpo_setup.py
```

This will test:
- TRL library imports
- RLAIF-V dataset loading
- LLaVA v1.6 model loading

### 2. Run Test Training

Run a quick test with 100 samples to ensure training works:

```bash
source venv/bin/activate
python orpo_llava_training_test.py
```

This will:
- Use only 100 training samples
- Train for maximum 50 steps
- Save model to `./checkpoints/llava-v1.6-7b-orpo-test`

### 3. Run Full Training

Once testing is successful, run the full training:

```bash
# Using the launcher script (recommended)
./run_orpo_training.sh

# Or directly
source venv/bin/activate
python orpo_llava_training.py
```

## Training Configuration

### Key Parameters

- **Model**: `liuhaotian/llava-v1.6-mistral-7b`
- **LoRA Rank**: 128 (full training), 64 (test)
- **LoRA Alpha**: 256 (full training), 128 (test)
- **Learning Rate**: 5e-5
- **Batch Size**: 2 per device with 32 gradient accumulation steps
- **Max Length**: 2048 tokens (full), 1024 (test)
- **ORPO Beta**: 0.1
- **Training Epochs**: 1

### Memory Optimization

- **LoRA**: Reduces trainable parameters from 7B to ~55M
- **bfloat16**: Reduces memory usage by half
- **Gradient Checkpointing**: Trades compute for memory
- **Image Resizing**: Limits images to 336px (224px for test)

## Dataset Information

The RLAIF-V dataset contains:
- **Total Samples**: 83,000+ preference pairs
- **Format**: Question, Image, Chosen Response, Rejected Response
- **Purpose**: Reduce hallucinations in vision-language models
- **Cache Location**: `/workspace/.cache/huggingface/datasets`

### Sample Data Structure
```python
{
    "question": "How many families?",
    "image": PIL.Image,
    "chosen": "The image shows a Union Organization table setup with 18,000 families.",
    "rejected": "The image does not provide any information about families."
}
```

## Expected Outcomes

After training, you should see:

1. **Reduced Hallucinations**: Model should provide more accurate visual descriptions
2. **Better Preference Alignment**: Model should prefer factual over speculative responses
3. **Improved AMBER Scores**: Similar improvements as shown in the TRL blog post

## Monitoring Training

The training logs will show:
- **Loss values**: Should generally decrease
- **ORPO metrics**: Preference accuracy and reward margins
- **Memory usage**: Monitor GPU memory consumption
- **Speed**: Steps per second and ETA

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**:
   - Reduce batch size to 1
   - Increase gradient accumulation steps
   - Use smaller image sizes
   - Enable gradient checkpointing

2. **TRL Import Errors**:
   - Ensure virtual environment is activated
   - Reinstall TRL: `pip install --upgrade trl`

3. **Dataset Loading Issues**:
   - Check internet connection
   - Verify cache directory permissions
   - Clear cache if corrupted: `rm -rf /workspace/.cache/huggingface/datasets`

4. **Model Loading Issues**:
   - Ensure sufficient disk space
   - Check model path and permissions
   - Verify cache directory

### Memory Estimation

For LLaVA v1.6 7B with LoRA:
- **Model**: ~14 GB (bfloat16)
- **LoRA adapters**: ~0.2 GB
- **Gradients**: ~0.2 GB
- **Optimizer states**: ~0.4 GB
- **Activations**: ~8-12 GB (depends on batch size)
- **Total**: ~23-27 GB

## Advanced Configuration

### Custom Model Path
```python
model_args.model_name_or_path = "path/to/your/model"
```

### Different LoRA Configuration
```python
peft_config = LoraConfig(
    r=256,  # Higher rank for more capacity
    lora_alpha=512,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
)
```

### Multi-GPU Training
```bash
accelerate launch --num_processes=4 orpo_llava_training.py
```

## Results and Evaluation

After training, evaluate your model using:
1. **AMBER benchmark** (if available)
2. **Manual inspection** of generated responses
3. **Comparison** with base model outputs

## References

- [TRL ORPO Documentation](https://huggingface.co/docs/trl/main/en/orpo_trainer)
- [RLAIF-V Dataset](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset)
- [LLaVA v1.6 Model](https://huggingface.co/liuhaotian/llava-v1.6-mistral-7b)
- [ORPO Paper](https://arxiv.org/abs/2403.07691)

## Support

If you encounter issues:
1. Check the logs for specific error messages
2. Verify all prerequisites are met
3. Try the test version first
4. Monitor GPU memory usage

Happy training! 🚀 