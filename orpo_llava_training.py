#!/usr/bin/env python3
"""
ORPO Training Script for LLaVA v1.6 7B on RLAIF-V Dataset
"""

import os
import torch
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset, features
from transformers import AutoTokenizer, TrainingArguments
from trl import ORPOConfig, ORPOTrainer
from peft import LoraConfig, get_peft_model
from llava.model import LlavaLlamaForCausalLM
from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up workspace cache for datasets
os.environ['HF_DATASETS_CACHE'] = '/workspace/.cache/huggingface/datasets'

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="liuhaotian/llava-v1.6-mistral-7b")
    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default="openai/clip-vit-large-patch14-336")
    mm_vision_select_layer: Optional[int] = field(default=-2)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)

@dataclass
class DataArguments:
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_aspect_ratio: str = 'pad'
    image_processor: Optional[object] = None

def format_dataset_for_orpo(dataset, tokenizer, image_processor):
    """Format RLAIF-V dataset for ORPO training"""
    def format_example(example):
        # Build conversation
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
            max_size = min(image_processor.size["longest_edge"], 336)  # Limit to 336 for memory
        else:
            max_size = 336
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
        num_proc=4,
        desc="Formatting dataset"
    )
    
    # Ensure images are properly decoded
    f = formatted_dataset.features
    f["images"] = features.Sequence(features.Image(decode=True))
    formatted_dataset = formatted_dataset.cast(f)
    
    return formatted_dataset

def main():
    # Load model and tokenizer
    model_args = ModelArguments()
    data_args = DataArguments()
    
    logger.info(f"Loading model: {model_args.model_name_or_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir="/workspace/.cache/huggingface/transformers",
        model_max_length=2048,
        padding_side="right",
        use_fast=False,
    )
    
    # Load model with proper configuration
    model_name = model_args.model_name_or_path or "liuhaotian/llava-v1.6-mistral-7b"
    if 'mistral' in model_name.lower():
        model = LlavaMistralForCausalLM.from_pretrained(
            model_name,
            cache_dir="/workspace/.cache/huggingface/transformers",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    else:
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_name,
            cache_dir="/workspace/.cache/huggingface/transformers",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    
    model.config.use_cache = False
    
    # Initialize vision modules
    if hasattr(model, 'get_model'):
        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(dtype=torch.bfloat16)
        image_processor = vision_tower.image_processor
        data_args.image_processor = image_processor
        data_args.is_multimodal = True
        
        # Configure vision settings
        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        
        # Set up multimodal tokens
        model.config.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_TOKEN], special_tokens=True)
        if model_args.mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        
        model.resize_token_embeddings(len(tokenizer))
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    
    # Configure LoRA
    peft_config = LoraConfig(
        r=128,
        lora_alpha=256,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Load RLAIF-V dataset
    logger.info("Loading RLAIF-V dataset...")
    dataset = load_dataset(
        "openbmb/RLAIF-V-Dataset", 
        split="train",
        cache_dir="/workspace/.cache/huggingface/datasets"
    )
    
    # Use a subset for testing (remove this for full training)
    # dataset = dataset.select(range(1000))  # Comment out for full dataset
    
    # Format dataset for ORPO
    train_dataset = format_dataset_for_orpo(dataset, tokenizer, image_processor)
    
    # Training configuration
    training_args = ORPOConfig(
        output_dir="./checkpoints/llava-v1.6-7b-orpo-rlaif-v",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=32,
        per_device_eval_batch_size=2,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        learning_rate=5e-5,
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        bf16=True,
        logging_steps=10,
        report_to="none",  # Change to "wandb" if you want to use wandb
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        max_length=2048,
        max_prompt_length=1024,
        beta=0.1,  # ORPO beta parameter
        dataset_num_proc=4,
    )
    
    # Initialize ORPO trainer
    trainer = ORPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    # Start training
    logger.info("Starting ORPO training...")
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    logger.info(f"Training completed! Model saved to {training_args.output_dir}")

if __name__ == "__main__":
    main() 