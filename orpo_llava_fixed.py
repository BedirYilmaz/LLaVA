#!/usr/bin/env python3
"""
Fixed ORPO Training Script for LLaVA v1.6 7B on RLAIF-V Dataset
This version avoids expanding the tokenizer vocabulary to prevent size mismatches.
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

def format_dataset_for_orpo(dataset, tokenizer, image_processor):
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
        
        # Process image - resize to prevent OOM
        image = example["image"]
        if hasattr(image_processor, 'size') and 'longest_edge' in image_processor.size:
            max_size = min(image_processor.size.get("longest_edge", 336), 224)
        else:
            max_size = 224
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
        num_proc=2,
        desc="Formatting dataset"
    )
    
    # Ensure images are properly decoded
    f = formatted_dataset.features
    f["images"] = features.Sequence(features.Image(decode=True))
    formatted_dataset = formatted_dataset.cast(f)
    
    return formatted_dataset

def main():
    model_args = ModelArguments()
    data_args = DataArguments()
    
    logger.info(f"Loading model: {model_args.model_name_or_path}")
    
    # Load tokenizer WITHOUT expanding vocabulary
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir="/workspace/.cache/huggingface/transformers",
        model_max_length=1024,
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
    
    # Configure LoRA
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Load RLAIF-V dataset (small subset for testing)
    logger.info("Loading RLAIF-V dataset (test subset)...")
    dataset = load_dataset(
        "openbmb/RLAIF-V-Dataset", 
        split="train[:50]",  # Very small for testing
        cache_dir="/workspace/.cache/huggingface/datasets"
    )
    
    logger.info(f"Using test dataset with {len(dataset)} samples")
    
    # Format dataset for ORPO
    train_dataset = format_dataset_for_orpo(dataset, tokenizer, image_processor)
    
    # Training configuration
    training_args = ORPOConfig(
        output_dir="./checkpoints/llava-v1.6-7b-orpo-fixed",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=2,
        learning_rate=5e-5,
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        bf16=True,
        logging_steps=1,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        max_length=1024,
        max_prompt_length=512,
        beta=0.1,
        dataset_num_proc=2,
        max_steps=20,  # Very limited for testing
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
        logger.info("🚀 Starting ORPO training (FIXED VERSION - NO VOCAB EXPANSION)...")
        
        # Start training
        trainer.train()
        
        # Save the final model
        trainer.save_model()
        logger.info(f"✅ Training completed! Model saved to {training_args.output_dir}")
        
        # Verify final tokenizer size
        final_vocab_size = len(tokenizer)
        if final_vocab_size == original_vocab_size:
            logger.info(f"✅ SUCCESS: Tokenizer vocabulary unchanged ({original_vocab_size} -> {final_vocab_size})")
        else:
            logger.error(f"❌ FAILURE: Tokenizer vocabulary changed ({original_vocab_size} -> {final_vocab_size})")
            
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 