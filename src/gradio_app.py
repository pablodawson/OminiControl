import gradio as gr
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers.pipelines import FluxPipeline
from diffusers import FluxTransformer2DModel
import numpy as np

from flux.condition import Condition
from flux.generate import seed_everything, generate

pipe = None
use_int8 = False

def get_gpu_memory():
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


def init_pipeline():
    global pipe
    if use_int8 or get_gpu_memory() < 33:
        transformer_model = FluxTransformer2DModel.from_pretrained(
            "sayakpaul/flux.1-schell-int8wo-improved",
            torch_dtype=torch.bfloat16,
            use_safetensors=False,
        )
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            transformer=transformer_model,
            torch_dtype=torch.bfloat16,
        )
    else:
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, #cache_dir ="/workspace/models"
        )
    pipe = pipe.to("cuda")
    pipe.load_lora_weights(
        "pablodawson/hm-omini",
        adapter_name="tryon",
    )


def process_image_and_text(image, text, target_width=512, condition_scale=1.0, steps=40, num_outputs=1):
    
    if image == None:
        return None, "Sube una imagen para continuar"

    if text == "":
        return None, "Escribe un texto para continuar"

    aspect_ratio = 1.5
    # Calculate target height based on aspect ratio
    target_height = int(target_width * aspect_ratio)
    
    # Get current dimensions
    w, h = image.size
    
    # Calculate dimensions to maintain aspect ratio with padding
    if h/w < aspect_ratio:
        # Image is wider than target ratio
        new_h = int(w * aspect_ratio)
        padding = (new_h - h) // 2
        padded = Image.new('RGB', (w, new_h), (255, 255, 255))
        padded.paste(image, (0, padding))
        image = padded
    else:
        # Image is taller than target ratio
        new_w = int(h / aspect_ratio)
        padding = (new_w - w) // 2
        padded = Image.new('RGB', (new_w, h), (255, 255, 255))
        padded.paste(image, (padding, 0))
        image = padded
    
    target_width = target_width // 16 * 16
    target_height = target_height // 16 * 16
    
    # Crop and resize
    image = image.resize((target_width, target_height))

    condition = Condition("tryon", image, position_delta=(0, target_width//16))

    if pipe is None:
        init_pipeline()

    results = []
    
    for i in range(num_outputs):
        seed_everything(i*10)
        
        result_img = generate(
            pipe,
            prompt=text.strip(),
            conditions=[condition],
            num_inference_steps=steps,
            height=target_height,
            condition_scale=condition_scale,
            width=target_width,
        ).images[0]

        results.append(result_img)

    return results, "Listo"


demo = gr.Interface(
    fn=process_image_and_text,
    inputs=[
        gr.Image(type="pil"),
        gr.Textbox(lines=2),
        gr.Slider(512, 1024, 512, 64, label="Width (ancho de la imagen)"),
        gr.Slider(0.5, 2.0, 1.0, 0.1, label="Condition Scale (que tanto ocupar la imagen de referencia)"),
        gr.Slider(20, 80, 40, 5, label="Steps (mas steps = mas tiempo pero mejor calidad)"),
        gr.Slider(1, 5, 1, 1, label="Cantidad de imagenes a generar")
    ],
    outputs=[gr.Gallery(), gr.Label()],
    title="KaysV2"
)

if __name__ == "__main__":
    init_pipeline()
    demo.launch(
        debug=False,
        #share=True
    )
