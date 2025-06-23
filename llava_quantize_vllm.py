import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.model.builder import load_pretrained_model

def main():
    # Model and LoRA paths
    MODEL_ID = "liuhaotian/llava-v1.6-mistral-7b"
    LORA_PATH = "/workspace/LLaVA/checkpoints/llava_v1.6-7b-orpo-production_bs_20/checkpoint-1560"

    # Load base model and tokenizer
    print(f"Loading base model: {MODEL_ID}")
    model_name = get_model_name_from_path(MODEL_ID)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        MODEL_ID, 
        None, 
        model_name,
        load_8bit=False,
        load_4bit=False,
        device='cuda'
    )
    print(f"✅ Model loaded successfully! Context length: {context_len}")

    if LORA_PATH:
        model = PeftModel.from_pretrained(model, LORA_PATH)
        model = model.merge_and_unload()
        print(f"✅ LoRA adapter loaded successfully! Context length: {context_len}")

    # Quantization recipe: FP8_DYNAMIC for all Linear layers except lm_head
    print("Applying FP8 quantization...")
    recipe = QuantizationModifier(
        targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]
    )
    oneshot(model=model, recipe=recipe)
    print("Quantization complete.")

    quantization_folder = "quantized_models"
    os.makedirs(quantization_folder, exist_ok=True)

    # Save quantized model and tokenizer
    save_dir = quantization_folder + "/" + MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-Dynamic"
    print(f"Saving quantized model to: {save_dir}")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir, safe_serialization=True)
    print("Done.")

if __name__ == "__main__":
    main()