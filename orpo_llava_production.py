#!/usr/bin/env python3
"""
Production ORPO Training Script for LLaVA v1.6 7B on RLAIF-V Dataset
This version avoids expanding the tokenizer vocabulary to prevent size mismatches.
Configured for full-scale training.
"""

import os
import torch
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset, features
from transformers import AutoTokenizer
from trl import ORPOConfig, ORPOTrainer
from peft import LoraConfig
from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
import logging
import argparse
import wandb

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up workspace cache for datasets
os.environ['HF_DATASETS_CACHE'] = '/workspace/.cache/huggingface/datasets'

@dataclass
class ModelArguments:
    model_name_or_path: str = "liuhaotian/llava-v1.6-mistral-7b"
    mm_use_im_start_end: bool = False
    mm_use_im_patch_token: bool = True

@dataclass
class DataArguments:
    image_aspect_ratio: str = 'pad'
    max_image_size: int = 336  # Full resolution for production

@dataclass 
class TrainingArguments:
    # Dataset settings
    use_full_dataset: bool = True
    dataset_subset_size: Optional[int] = None  # None = full dataset
    
    # Training settings
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    max_length: int = 2048  # Full length for production
    max_prompt_length: int = 1024
    
    # LoRA settings
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    
    # Saving and logging
    save_steps: int = 500
    logging_steps: int = 10
    output_dir: str = "./checkpoints/llava-v1.6-7b-orpo-production"

def format_dataset_for_orpo(dataset, tokenizer, image_processor, data_args):
    """Format RLAIF-V dataset for ORPO training without expanding tokenizer vocabulary"""
    
    def format_example(example):
        # Build conversation using vicuna_v1 template
        conv = conv_templates["vicuna_v1"].copy()
        
        # Add the image token and question
        if DEFAULT_IMAGE_TOKEN not in example["question"]:
            question = DEFAULT_IMAGE_TOKEN + '\n' + example["question"]
        else:
            question = example["question"]
        
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # Create chosen and rejected responses
        chosen_response = example["chosen"]
        rejected_response = example["rejected"]
        
        # Process image - use full resolution for production
        image = example["image"]
        max_size = data_args.max_image_size
        
        # Only resize if image is larger than max_size
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size))
        
        return {
            "images": [image],
            "prompt": prompt,
            "chosen": chosen_response,
            "rejected": rejected_response,
        }
    
    logger.info("Formatting dataset for ORPO...")
    formatted_dataset = dataset.map(
        format_example, 
        remove_columns=dataset.column_names,
        num_proc=8,  # More processes for production
        desc="Formatting dataset"
    )
    
    # Ensure images are properly decoded
    f = formatted_dataset.features
    f["images"] = features.Sequence(features.Image(decode=True))
    formatted_dataset = formatted_dataset.cast(f)
    
    return formatted_dataset

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_mode", action="store_true", help="Run in test mode with small dataset")
    parser.add_argument("--dataset_size", type=int, help="Limit dataset size (for testing)")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Per device batch size")
    parser.add_argument("--wandb_project", type=str, default="llava-orpo", help="wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="wandb entity (team/user)")
    args = parser.parse_args()
    
    # Initialize configurations
    model_args = ModelArguments()
    data_args = DataArguments()
    train_args = TrainingArguments()
    
    # Override with command line arguments
    if args.test_mode:
        train_args.dataset_subset_size = 100
        train_args.num_train_epochs = 1
        train_args.save_steps = 10
        train_args.output_dir = "./checkpoints/llava-v1.6-7b-orpo-test"
        logger.info("🧪 Running in TEST MODE")
    
    if args.dataset_size:
        train_args.dataset_subset_size = args.dataset_size
    if args.output_dir:
        train_args.output_dir = args.output_dir
    if args.epochs:
        train_args.num_train_epochs = args.epochs
    if args.batch_size:
        train_args.per_device_train_batch_size = args.batch_size
    
    logger.info(f"Loading model: {model_args.model_name_or_path}")
    
    # Load tokenizer WITHOUT expanding vocabulary
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir="/workspace/.cache/huggingface/transformers",
        model_max_length=train_args.max_length,
        padding_side="right",
        use_fast=False,
    )
    
    original_vocab_size = len(tokenizer)
    logger.info(f"Original tokenizer vocabulary size: {original_vocab_size}")
    
    # Load model
    model = LlavaMistralForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir="/workspace/.cache/huggingface/transformers",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    
    # Handle tuple return if needed
    if isinstance(model, tuple):
        model = model[0]
    
    model.config.use_cache = False
    
    # Initialize vision modules
    if hasattr(model, 'get_vision_tower'):
        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(dtype=torch.bfloat16)
        image_processor = vision_tower.image_processor
        
        # Configure vision settings
        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        
        # Set up multimodal tokens configuration (but DON'T add to tokenizer)
        model.config.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        
        # CRITICAL: Do NOT add tokens to tokenizer or resize embeddings
        # The tokenizer_image_token function handles <image> replacement with IMAGE_TOKEN_INDEX (-200)
        logger.info("✅ IMPORTANT: Skipping tokenizer vocabulary expansion to prevent size mismatch!")
        logger.info(f"✅ Using IMAGE_TOKEN_INDEX ({IMAGE_TOKEN_INDEX}) for <image> token replacement")
    
    # Verify embedding size matches tokenizer
    embedding_size = model.get_input_embeddings().weight.shape[0]
    logger.info(f"Model embedding size: {embedding_size}")
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
    
    if embedding_size != len(tokenizer):
        logger.error(f"❌ SIZE MISMATCH: embedding {embedding_size} != vocab {len(tokenizer)}")
        return
    else:
        logger.info("✅ SUCCESS: Model embedding size matches tokenizer vocabulary size!")
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    
    # Configure LoRA for production
    peft_config = LoraConfig(
        r=train_args.lora_r,
        lora_alpha=train_args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=train_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Load RLAIF-V dataset
    if train_args.dataset_subset_size:
        logger.info(f"Loading RLAIF-V dataset (subset of {train_args.dataset_subset_size} samples)...")
        dataset = load_dataset(
            "openbmb/RLAIF-V-Dataset", 
            split=f"train[:{train_args.dataset_subset_size}]",
            cache_dir="/workspace/.cache/huggingface/datasets"
        )
    else:
        logger.info("Loading RLAIF-V dataset (FULL DATASET)...")
        dataset = load_dataset(
            "openbmb/RLAIF-V-Dataset", 
            split="train",
            cache_dir="/workspace/.cache/huggingface/datasets"
        )
    
    logger.info(f"Using dataset with {len(dataset)} samples")
    
    # Format dataset for ORPO
    train_dataset = format_dataset_for_orpo(dataset, tokenizer, image_processor, data_args)
    
    # Initialize wandb
    wandb_run = None
    if args.wandb_project:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"llava-v1.6-7b-orpo-prod-epochs{train_args.num_train_epochs}-bs{train_args.per_device_train_batch_size}",
            config={
                "model_name_or_path": model_args.model_name_or_path,
                "num_train_epochs": train_args.num_train_epochs,
                "per_device_train_batch_size": train_args.per_device_train_batch_size,
                "gradient_accumulation_steps": train_args.gradient_accumulation_steps,
                "learning_rate": train_args.learning_rate,
                "max_length": train_args.max_length,
                "max_prompt_length": train_args.max_prompt_length,
                "lora_r": train_args.lora_r,
                "lora_alpha": train_args.lora_alpha,
                "lora_dropout": train_args.lora_dropout,
                "output_dir": train_args.output_dir,
                "dataset_subset_size": train_args.dataset_subset_size,
                "image_aspect_ratio": data_args.image_aspect_ratio,
                "max_image_size": data_args.max_image_size,
            }
        )
        logger.info(f"✅ wandb initialized: project={args.wandb_project}, entity={args.wandb_entity}")
    
    # Training configuration for production
    training_args = ORPOConfig(
        output_dir=train_args.output_dir,
        num_train_epochs=train_args.num_train_epochs,
        per_device_train_batch_size=train_args.per_device_train_batch_size,
        gradient_accumulation_steps=train_args.gradient_accumulation_steps,
        save_strategy="steps",
        save_steps=train_args.save_steps,
        save_total_limit=3,
        learning_rate=train_args.learning_rate,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=train_args.logging_steps,
        gradient_checkpointing=True,
        dataloader_num_workers=8,  # More workers for production
        remove_unused_columns=False,
        max_length=train_args.max_length,
        max_prompt_length=train_args.max_prompt_length,
        beta=0.1,  # ORPO beta parameter
        dataset_num_proc=8,  # More processes for production
        report_to=["wandb"],
    )
    
    # Initialize ORPO trainer
    try:
        trainer = ORPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
        
        logger.info("✅ ORPO Trainer initialized successfully!")
        logger.info("🚀 Starting ORPO training (PRODUCTION VERSION - NO VOCAB EXPANSION)...")
        logger.info(f"📊 Dataset size: {len(dataset)} samples")
        logger.info(f"🏋️ Training epochs: {train_args.num_train_epochs}")
        logger.info(f"📦 Batch size: {train_args.per_device_train_batch_size}")
        logger.info(f"📁 Output directory: {train_args.output_dir}")
        
        # Start training
        trainer.train()
        
        # Save the final model
        trainer.save_model()
        logger.info(f"✅ Training completed! Model saved to {train_args.output_dir}")
        
        # Verify final tokenizer size
        final_vocab_size = len(tokenizer)
        if final_vocab_size == original_vocab_size:
            logger.info(f"✅ SUCCESS: Tokenizer vocabulary unchanged ({original_vocab_size} -> {final_vocab_size})")
        else:
            logger.error(f"❌ FAILURE: Tokenizer vocabulary changed ({original_vocab_size} -> {final_vocab_size})")
            
        # Log final metrics to wandb
        if wandb_run is not None:
            wandb_run.log({"final_epoch": train_args.num_train_epochs, "final_batch_size": train_args.per_device_train_batch_size})
            wandb_run.finish()
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 