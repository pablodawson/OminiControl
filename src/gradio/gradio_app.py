import gradio as gr
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers.pipelines import FluxPipeline
from diffusers import FluxTransformer2DModel
import numpy as np

from ..flux.condition import Condition
from ..flux.generate import seed_everything, generate

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
            "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
        )
    pipe = pipe.to("cuda")
    pipe.load_lora_weights(
        "Yuanshi/OminiControl",
        weight_name="/workspace1/pdawson/OminiControl/runs/20250223-211749/ckpt/15000/pytorch_lora_weights.safetensors",
        adapter_name="tryon",
    )


def process_image_and_text(image, text, target_width=512, condition_scale=1.0, aspect_ratio=1.5):
    # Calculate target height based on aspect ratio
    target_height = int(target_width / aspect_ratio)
    
    # Get current dimensions
    w, h = image.size
    
    # Calculate dimensions to maintain aspect ratio with padding
    if w/h > aspect_ratio:
        # Image is wider than target ratio
        new_h = int(w / aspect_ratio)
        padding = (new_h - h) // 2
        padded = Image.new('RGB', (w, new_h), (255, 255, 255))
        padded.paste(image, (0, padding))
        image = padded
    else:
        # Image is taller than target ratio
        new_w = int(h * aspect_ratio)
        padding = (new_w - w) // 2
        padded = Image.new('RGB', (new_w, h), (255, 255, 255))
        padded.paste(image, (padding, 0))
        image = padded
    
    # Crop and resize
    image = image.resize((target_width, target_height))

    condition = Condition("subject", image, position_delta=(0, target_width//16))

    if pipe is None:
        init_pipeline()

    result_img = generate(
        pipe,
        prompt=text.strip(),
        conditions=[condition],
        num_inference_steps=8,
        height=target_height,
        condition_scale=condition_scale,
        width=target_width,
    ).images[0]

    return result_img


def get_samples():
    sample_list = [
        {
            "image": "assets/oranges.jpg",
            "text": "A very close up view of this item. It is placed on a wooden table. The background is a dark room, the TV is on, and the screen is showing a cooking show. With text on the screen that reads 'Omini Control!'",
        },
        {
            "image": "assets/penguin.jpg",
            "text": "On Christmas evening, on a crowded sidewalk, this item sits on the road, covered in snow and wearing a Christmas hat, holding a sign that reads 'Omini Control!'",
        },
        {
            "image": "assets/rc_car.jpg",
            "text": "A film style shot. On the moon, this item drives across the moon surface. The background is that Earth looms large in the foreground.",
        },
        {
            "image": "assets/clock.jpg",
            "text": "In a Bauhaus style room, this item is placed on a shiny glass table, with a vase of flowers next to it. In the afternoon sun, the shadows of the blinds are cast on the wall.",
        },
        {
            "image": "assets/tshirt.jpg",
            "text": "On the beach, a lady sits under a beach umbrella with 'Omini' written on it. She's wearing this shirt and has a big smile on her face, with her surfboard hehind her.",
        },
    ]
    return [[Image.open(sample["image"]), sample["text"]] for sample in sample_list]


demo = gr.Interface(
    fn=process_image_and_text,
    inputs=[
        gr.Image(type="pil"),
        gr.Textbox(lines=2),
        gr.Slider(0.5, 2.0, 0.1, default=1.0, label="Condition Scale"),
        gr.Slider(512, 1024, 64, default=512, label="Target Width"),
    ],
    outputs=gr.Image(type="pil"),
    title="OminiControl / Subject driven generation",
    examples=get_samples(),
)

if __name__ == "__main__":
    init_pipeline()
    demo.launch(
        debug=True,
    )
