import os
import sys
import torch
from PIL import Image
import gradio as gr

# Add LLaVA to Python path
sys.path.append('/workspace/LLaVA')

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from peft import PeftModel

# ---- Model Loading ----
def load_llava_model(model_path="liuhaotian/llava-v1.6-mistral-7b", model_base=None, device='cuda', load_8bit=False, load_4bit=False, lora_path=None):
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

# ---- Input Preparation ----
def prepare_llava_input(prompt, image, tokenizer, image_processor, model_config):
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    if DEFAULT_IMAGE_TOKEN not in prompt:
        if getattr(model_config, 'mm_use_im_start_end', False):
            prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    image_tensor = process_images([image], image_processor, model_config)
    if isinstance(image_tensor, list):
        image_tensor = [img.to(dtype=torch.float16) for img in image_tensor]
    else:
        image_tensor = image_tensor.to(dtype=torch.float16)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    if hasattr(input_ids, 'unsqueeze'):
        input_ids = input_ids.unsqueeze(0)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    if not stop_str or stop_str.isspace():
        stop_str = "</s>"
    return input_ids, image_tensor, stop_str

# ---- Inference ----
def generate_response(model, tokenizer, input_ids, image_tensor, stop_str, image_sizes, max_new_tokens=512, temperature=0.7):
    device = model.device
    input_ids = input_ids.to(device)
    if isinstance(image_tensor, list):
        image_tensor = [img.to(device) for img in image_tensor]
    else:
        image_tensor = image_tensor.to(device)
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    input_token_len = input_ids.shape[1]
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs

# ---- Gradio Inference Wrapper ----
def gradio_infer(image, prompt):
    try:
        if image is None or prompt.strip() == "":
            return "Please provide both an image and a prompt."
        pil_image = image if isinstance(image, Image.Image) else Image.fromarray(image)
        input_ids, image_tensor, stop_str = prepare_llava_input(
            prompt, pil_image, gradio_infer.tokenizer, gradio_infer.image_processor, gradio_infer.model.config
        )
        image_sizes = [pil_image.size]
        response = generate_response(
            gradio_infer.model, gradio_infer.tokenizer, input_ids, image_tensor, stop_str, image_sizes,
            max_new_tokens=512, temperature=0.7
        )
        return response
    except Exception as e:
        return f"Error: {e}"

# ---- Load model at startup ----
MODEL_PATH = os.environ.get("LLAVA_MODEL_PATH", "liuhaotian/llava-v1.6-mistral-7b")
LORA_PATH = os.environ.get("LLAVA_LORA_PATH", None)
print(f"Loading model: {MODEL_PATH}, LoRA: {LORA_PATH}")
tokenizer, model, image_processor, context_len = load_llava_model(MODEL_PATH, lora_path=LORA_PATH)
gradio_infer.tokenizer = tokenizer
gradio_infer.model = model
gradio_infer.image_processor = image_processor

# ---- Gradio UI ----
description = """
# LLaVA with LoRA Gradio Demo\n
Upload an image and enter a prompt. The model will generate a response conditioned on both.
"""

demo = gr.Interface(
    fn=gradio_infer,
    inputs=[
        gr.Image(type="pil", label="Image"),
        gr.Textbox(lines=2, label="Prompt")
    ],
    outputs=gr.Textbox(label="Response"),
    title="LLaVA with LoRA Gradio Demo",
    description=description,
    allow_flagging="never"
)

def main():
    demo.launch()

if __name__ == "__main__":
    main() 