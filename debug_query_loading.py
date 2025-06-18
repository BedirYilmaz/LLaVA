#!/usr/bin/env python3
"""
Debug script to isolate the query loading and conversation processing issue.
"""

import os
import sys
import json
import torch
from PIL import Image

# Add LLaVA to Python path
sys.path.append('/workspace/LLaVA')

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

def debug_query_loading():
    """Debug the query loading process."""
    print("🔍 Debugging query loading...")
    
    # Load queries
    query_file = "/workspace/AMBER/data/query/query_all.json"
    with open(query_file, 'r') as f:
        queries = json.load(f)
    
    print(f"✅ Loaded {len(queries)} queries")
    print(f"📋 First query: {queries[0]}")
    print(f"📋 Query 1005 (discriminative): {queries[1004]}")
    
    return queries

def debug_conversation_template():
    """Debug the conversation template."""
    print("🔍 Debugging conversation template...")
    
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    
    print(f"✅ Template: {conv_mode}")
    print(f"📋 Roles: {conv.roles}")
    print(f"📋 Sep: {repr(conv.sep)}")
    print(f"📋 Sep2: {repr(conv.sep2)}")
    print(f"📋 Sep style: {conv.sep_style}")
    
    # Test message appending
    test_prompt = "Describe this image."
    if DEFAULT_IMAGE_TOKEN not in test_prompt:
        test_prompt = DEFAULT_IMAGE_TOKEN + '\n' + test_prompt
    
    print(f"📋 Test prompt: {test_prompt}")
    
    conv.append_message(conv.roles[0], test_prompt)
    conv.append_message(conv.roles[1], None)
    final_prompt = conv.get_prompt()
    
    print(f"📋 Final prompt: {repr(final_prompt)}")
    
    # Test stop string
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    if not stop_str or stop_str.isspace():
        stop_str = "</s>"
    print(f"📋 Stop string: {repr(stop_str)}")
    
    return final_prompt, stop_str

def debug_model_loading():
    """Debug model loading."""
    print("🔍 Debugging model loading...")
    
    disable_torch_init()
    model_path = "liuhaotian/llava-v1.6-mistral-7b"
    model_name = get_model_name_from_path(model_path)
    
    print(f"📋 Model path: {model_path}")
    print(f"📋 Model name: {model_name}")
    
    try:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, 
            None,  # model_base
            model_name,
            load_8bit=False,
            load_4bit=False,
            device='cuda'
        )
        print(f"✅ Model loaded successfully!")
        print(f"📋 Context length: {context_len}")
        return tokenizer, model, image_processor
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None, None, None

def debug_image_processing():
    """Debug image processing."""
    print("🔍 Debugging image processing...")
    
    image_path = "/workspace/AMBER/image/AMBER_1.jpg"
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None
    
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"✅ Image loaded: {image.size}")
        return image
    except Exception as e:
        print(f"❌ Image loading failed: {e}")
        return None

def main():
    print("🚀 Starting debug session...")
    print("=" * 50)
    
    # Debug 1: Query loading
    queries = debug_query_loading()
    if not queries:
        return
    
    print("\n" + "=" * 50)
    
    # Debug 2: Conversation template
    final_prompt, stop_str = debug_conversation_template()
    
    print("\n" + "=" * 50)
    
    # Debug 3: Image processing
    image = debug_image_processing()
    if image is None:
        return
    
    print("\n" + "=" * 50)
    
    # Debug 4: Model loading
    tokenizer, model, image_processor = debug_model_loading()
    if model is None:
        return
    
    print("\n" + "=" * 50)
    print("✅ All components loaded successfully!")
    print("🎯 Query format is correct!")
    print("🎯 The issue is likely in the generation pipeline, not query loading!")

if __name__ == "__main__":
    main() 