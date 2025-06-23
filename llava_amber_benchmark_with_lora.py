#!/usr/bin/env python3
"""
LLaVA-v1.6-Mistral-7B AMBER Benchmark Script
This script uses the LLaVA repository's native model loading and runs the complete AMBER benchmark.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image
import subprocess
import time

from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training


# Add LLaVA to Python path
sys.path.append('/workspace/LLaVA')

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
print("✅ LLaVA modules imported successfully!")


def load_llava_model(model_path="liuhaotian/llava-v1.6-mistral-7b", model_base=None, device='cuda', load_8bit=False, load_4bit=False, lora_path=None):
    """Load LLaVA model using native codebase."""
    print(f"🔄 Loading LLaVA model from: {model_path}")
    
    disable_torch_init()
    
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, 
        model_base, 
        model_name,
        load_8bit=load_8bit,
        load_4bit=load_4bit,
        device=device
    )
    print(f"✅ Model loaded successfully! Context length: {context_len}")

    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
        print(f"✅ LoRA adapter loaded successfully! Context length: {context_len}")
    
    return tokenizer, model, image_processor, context_len


def get_conversation_template(model_config):
    """Get the appropriate conversation template for the model."""
    # Use the native LLaVA v1 conversation template
    conv_mode = "llava_v1"
    
    print(f"🎯 Using conversation template: {conv_mode}")
    return conv_mode


def prepare_llava_input(prompt, image, tokenizer, image_processor, model_config):
    """Prepare input for LLaVA model."""
    conv_mode = get_conversation_template(model_config)
    conv = conv_templates[conv_mode].copy()
    
    # Add image token to prompt if not present
    if DEFAULT_IMAGE_TOKEN not in prompt:
        if model_config.mm_use_im_start_end:
            prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
    
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    # Process image
    image_tensor = process_images([image], image_processor, model_config)
    if type(image_tensor) is list:
        image_tensor = [image.to(dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(dtype=torch.float16)
    
    # Tokenize
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
    
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    # Handle empty or whitespace-only separators
    if not stop_str or stop_str.isspace():
        stop_str = "</s>"
    return input_ids, image_tensor, stop_str


def generate_response(model, tokenizer, input_ids, image_tensor, stop_str, image_sizes, max_new_tokens=512, temperature=0.7):
    """Generate response using LLaVA model."""
    device = model.device
    
    # Move inputs to device
    input_ids = input_ids.to(device)
    if type(image_tensor) is list:
        image_tensor = [image.to(device) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(device)
    
    # Set up stopping criteria
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,  # Pass image_sizes parameter
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode output
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    
    return outputs


def run_amber_benchmark(model_path, query_file, image_dir, output_file, max_samples=None, start_from=0, lora_path=None):
    """Run AMBER benchmark on given query file."""
    
    print(f"🚀 Starting AMBER benchmark")
    print(f"📋 Query file: {query_file}")
    print(f"🖼️  Image directory: {image_dir}")
    print(f"💾 Output file: {output_file}")
    print(f"📊 Max samples: {max_samples}")
    print(f"🎯 Starting from: {start_from}")
    
    # lora_path is now passed as an argument
    tokenizer, model, image_processor, context_len = load_llava_model(model_path, lora_path=lora_path)
    print("✅ Model loaded successfully!")
    
    # Load queries
    print("📄 Loading queries...")
    with open(query_file, 'r') as f:
        queries = json.load(f)
    
    # Filter queries
    if start_from > 0:
        queries = [q for q in queries if q['id'] >= start_from]
    
    if max_samples:
        queries = queries[:max_samples]
    
    print(f"📊 Processing {len(queries)} queries...")
    
    responses = []
    error_count = 0
    
    for query in tqdm(queries, desc="Processing queries"):
        query_id = query['id']
        
        # Load image
        image_path = os.path.join(image_dir, f"AMBER_{query_id}.jpg")
        if not os.path.exists(image_path):
            print(f"⚠️  Image not found: {image_path}")
            error_count += 1
            continue
            
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Determine task type and prepare prompt
            if query_id <= 1004:  # Generative task
                prompt = query['query']  # Use the query from the file
                max_tokens = 150
                temp = 0.7
            else:  # Discriminative task  
                prompt = f"{query['query']} Answer with only 'Yes' or 'No'."
                max_tokens = 10
                temp = 0.0
            
            # Prepare input
            input_ids, image_tensor, stop_str = prepare_llava_input(
                prompt, image, tokenizer, image_processor, model.config
            )
            
            # Generate response  
            image_sizes = [image.size]  # Get original image size
            response = generate_response(
                model, tokenizer, input_ids, image_tensor, stop_str, image_sizes,
                max_new_tokens=max_tokens, temperature=temp
            )
            
            responses.append({
                "id": query_id,
                "response": response
            })
            
        except Exception as e:
            print(f"❌ Error processing query {query_id}: {e}")
            error_count += 1
            responses.append({
                "id": query_id,
                "response": "Error processing image"
            })
            continue
    
    # Save responses
    print(f"💾 Saving {len(responses)} responses to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(responses, f, indent=2)
    
    print(f"✅ Benchmark completed!")
    print(f"📊 Processed: {len(responses)} queries")
    print(f"❌ Errors: {error_count}")
    
    return responses


def run_evaluation(inference_file, amber_root="/workspace/AMBER"):
    """Run AMBER evaluation on the generated responses."""
    print("🔍 Running AMBER evaluation...")
    
    # Get absolute path of inference file
    if not os.path.isabs(inference_file):
        inference_file = os.path.abspath(inference_file)
    
    # Change to AMBER directory
    original_dir = os.getcwd()
    os.chdir(amber_root)
    
    # Run inference evaluation
    cmd = [
        "python", "inference.py",
        "--inference_data", inference_file,
        "--evaluation_type", "a"  # All tasks and dimensions
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Evaluation completed successfully!")
        print("📊 Results:")
        print(result.stdout)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed: {e}")
        print(f"stderr: {e.stderr}")
        return None
    finally:
        # Return to original directory
        os.chdir(original_dir)


def main():
    parser = argparse.ArgumentParser(description="Run LLaVA on AMBER benchmark")
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.6-mistral-7b",
                        help="Path to LLaVA model")
    parser.add_argument("--query_file", type=str, default="/workspace/AMBER/data/query/query_all.json",
                        help="Path to AMBER query file")
    parser.add_argument("--image_dir", type=str, default="/workspace/AMBER/image",
                        help="Path to AMBER images directory")
    parser.add_argument("--output_file", type=str, default="llava_amber_with_lora_responses.json",
                        help="Output file for responses")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process")
    parser.add_argument("--start_from", type=int, default=0,
                        help="Start from this query ID")
    parser.add_argument("--skip_inference", action="store_true",
                        help="Skip inference and only run evaluation")
    parser.add_argument("--skip_evaluation", action="store_true",
                        help="Skip evaluation and only run inference")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to LoRA adapter (optional)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else ".", exist_ok=True)
    
    if not args.skip_inference:
        # Run benchmark
        responses = run_amber_benchmark(
            args.model_path,
            args.query_file,
            args.image_dir,
            args.output_file,
            args.max_samples,
            args.start_from,
            args.lora_path
        )
    
    if not args.skip_evaluation:
        # Run evaluation
        run_evaluation(args.output_file)


if __name__ == "__main__":
    main() 