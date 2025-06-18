#!/usr/bin/env python3
"""
Debug script to step through the generation pipeline and find the exact NoneType error.
"""

import os
import sys
import json
import torch
from PIL import Image
import traceback

# Add LLaVA to Python path
sys.path.append('/workspace/LLaVA')

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

def debug_generation_pipeline():
    """Debug the complete generation pipeline step by step."""
    
    print("🔍 Starting generation pipeline debug...")
    
    # Load model
    print("1️⃣ Loading model...")
    disable_torch_init()
    model_path = "liuhaotian/llava-v1.6-mistral-7b"
    model_name = get_model_name_from_path(model_path)
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, load_8bit=False, load_4bit=False, device='cuda'
    )
    print("✅ Model loaded")
    
    # Load image
    print("2️⃣ Loading image...")
    image_path = "/workspace/AMBER/image/AMBER_1.jpg"
    image = Image.open(image_path).convert('RGB')
    print("✅ Image loaded")
    
    # Prepare conversation
    print("3️⃣ Preparing conversation...")
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    
    prompt = "Describe this image."
    if DEFAULT_IMAGE_TOKEN not in prompt:
        prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
    
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    if not stop_str or stop_str.isspace():
        stop_str = "</s>"
        
    print(f"✅ Conversation prepared, stop_str: {repr(stop_str)}")
    
    # Process image
    print("4️⃣ Processing image...")
    try:
        image_tensor = process_images([image], image_processor, model.config)
        print(f"✅ Image tensor type: {type(image_tensor)}")
        
        if type(image_tensor) is list:
            image_tensor = [img.to(dtype=torch.float16) for img in image_tensor]
            print(f"✅ Converted list of {len(image_tensor)} tensors to float16")
        else:
            image_tensor = image_tensor.to(dtype=torch.float16)
            print(f"✅ Converted tensor to float16, shape: {image_tensor.shape}")
            
    except Exception as e:
        print(f"❌ Image processing failed: {e}")
        traceback.print_exc()
        return
    
    # Tokenize
    print("5️⃣ Tokenizing...")
    try:
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        print(f"✅ Tokenized, input_ids shape: {input_ids.shape}")
        
        input_ids = input_ids.unsqueeze(0)
        print(f"✅ Unsqueezed, input_ids shape: {input_ids.shape}")
        
    except Exception as e:
        print(f"❌ Tokenization failed: {e}")
        traceback.print_exc()
        return
    
    # Move to device
    print("6️⃣ Moving to device...")
    try:
        device = model.device
        input_ids = input_ids.to(device)
        
        if type(image_tensor) is list:
            image_tensor = [img.to(device) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(device)
            
        print(f"✅ Moved to device: {device}")
        
    except Exception as e:
        print(f"❌ Device movement failed: {e}")
        traceback.print_exc()
        return
    
    # Set up stopping criteria
    print("7️⃣ Setting up stopping criteria...")
    try:
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        print("✅ Stopping criteria created")
        
    except Exception as e:
        print(f"❌ Stopping criteria failed: {e}")
        traceback.print_exc()
        return
    
    # Generate
    print("8️⃣ Generating...")
    try:
        image_sizes = [image.size]  # Add missing image_sizes
        print(f"✅ Image sizes: {image_sizes}")
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,  # Add image_sizes parameter
                do_sample=True,
                temperature=0.7,
                max_new_tokens=50,  # Small number for testing
                pad_token_id=tokenizer.eos_token_id,  # Set pad_token_id to avoid warnings
                eos_token_id=tokenizer.eos_token_id
            )
        
        print(f"✅ Generation completed, output shape: {output_ids.shape}")
        
        # Decode
        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        
        print(f"✅ Decoded output: {outputs}")
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        traceback.print_exc()
        return
    
    print("🎉 Generation pipeline completed successfully!")

if __name__ == "__main__":
    debug_generation_pipeline() 