import os
from diffusers import StableDiffusionInpaintPipeline

os.makedirs("diffusers-cache", exist_ok=True)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    cache_dir="diffusers-cache")

# Download LORA
if not os.path.exists("rosjf-05.safetensors"):
    os.system("wget https://item-swapper.s3.amazonaws.com/loras/rosjf-05.safetensors")

# Clone sdscripts
if not os.path.exists("sdscripts"):
    os.system("git clone https://github.com/kohya-ss/sd-scripts.git")
    os.system("mv sd-scripts sdscripts")