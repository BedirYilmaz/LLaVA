#!/usr/bin/env python3
"""
Test script to validate ORPO setup for LLaVA v1.6 7B
"""

import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from llava.model import LlavaLlamaForCausalLM
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up workspace cache for datasets
os.environ['HF_DATASETS_CACHE'] = '/workspace/.cache/huggingface/datasets'

def test_dataset_loading():
    """Test loading and accessing RLAIF-V dataset"""
    logger.info("Testing dataset loading...")
    
    try:
        dataset = load_dataset(
            "openbmb/RLAIF-V-Dataset", 
            split="train[:1%]",  # Load only 1% for testing
            cache_dir="/workspace/.cache/huggingface/datasets"
        )
        
        logger.info(f"Dataset loaded successfully! Size: {len(dataset)}")
        
        # Check first sample
        sample = dataset[0]
        logger.info(f"Sample keys: {sample.keys()}")
        logger.info(f"Question: {sample['question'][:100]}...")
        logger.info(f"Chosen: {sample['chosen'][:100]}...")
        logger.info(f"Rejected: {sample['rejected'][:100]}...")
        logger.info(f"Image shape: {sample['image'].size}")
        
        return True
    except Exception as e:
        logger.error(f"Dataset loading failed: {e}")
        return False

def test_model_loading():
    """Test loading LLaVA v1.6 7B model"""
    logger.info("Testing model loading...")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "liuhaotian/llava-v1.6-mistral-7b",
            cache_dir="/workspace/.cache/huggingface/transformers",
            model_max_length=2048,
            padding_side="right",
            use_fast=False,
        )
        logger.info("Tokenizer loaded successfully!")
        
        # Load model
        model = LlavaLlamaForCausalLM.from_pretrained(
            "liuhaotian/llava-v1.6-mistral-7b",
            cache_dir="/workspace/.cache/huggingface/transformers",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        logger.info("Model loaded successfully!")
        
        # Test vision tower
        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        logger.info("Vision tower loaded successfully!")
        
        return True
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return False

def test_trl_import():
    """Test TRL imports"""
    logger.info("Testing TRL imports...")
    
    try:
        from trl import ORPOConfig, ORPOTrainer
        logger.info("TRL imports successful!")
        return True
    except Exception as e:
        logger.error(f"TRL import failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("Starting ORPO setup validation...")
    
    tests = [
        ("TRL Import", test_trl_import),
        ("Dataset Loading", test_dataset_loading),
        ("Model Loading", test_model_loading),
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("\n✅ All tests passed! Ready for ORPO training.")
    else:
        logger.error("\n❌ Some tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    main() 