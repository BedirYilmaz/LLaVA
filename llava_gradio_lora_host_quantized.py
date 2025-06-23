import os
import sys
from PIL import Image
import gradio as gr
import argparse# vLLM imports
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset

# ---- Gradio Inference Wrapper ----
def gradio_infer(image, prompt):
    try:
        if image is None or prompt.strip() == "":
            return "Please provide both an image and a prompt."
        pil_image = image if isinstance(image, Image.Image) else Image.fromarray(image)
        # Convert to vllm ImageAsset
        image_asset = ImageAsset(pil_image)
        # Format prompt for vllm
        question = prompt.strip()
        formatted_prompt = f"<|image|><|begin_of_text|>{question}"
        # Prepare inputs for vllm
        inputs = {
            "prompt": formatted_prompt,
            "multi_modal_data": {
                "image": image_asset.pil_image.convert("RGB")
            },
        }
        sampling_params = gradio_infer.sampling_params
        outputs = gradio_infer.llm.generate(inputs, sampling_params=sampling_params)
        return outputs[0].outputs[0].text
    except Exception as e:
        return f"Error: {e}"

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
    parser = argparse.ArgumentParser(description="LLaVA Gradio LoRA Host (vLLM)")
    parser.add_argument('--model_path', type=str, default=os.environ.get("LLAVA_MODEL_PATH", "/workspace/LLaVA/quantized_models/llava-v1.6-mistral-7b-FP8-Dynamic"), help='Path to quantized model directory or HuggingFace repo')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='Tensor parallel size for vllm')
    parser.add_argument('--max_num_seqs', type=int, default=1, help='Max number of sequences for vllm')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--max_tokens', type=int, default=512, help='Maximum number of tokens to generate')
    args = parser.parse_args()

    print(f"Loading vLLM model: {args.model_path}")
    llm = LLM(model=args.model_path, max_num_seqs=args.max_num_seqs, enforce_eager=True, tensor_parallel_size=args.tensor_parallel_size)
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    gradio_infer.llm = llm
    gradio_infer.sampling_params = sampling_params
    demo.launch()

if __name__ == "__main__":
    main()