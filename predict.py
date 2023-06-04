''' StableDiffusion-v1 Predict Module '''

import os
from typing import List
import time
import torch
from diffusers import (
    StableDiffusionInpaintPipeline,
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    IPNDMScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler
)
import json
from utils import *

from PIL import Image
from cog import BasePredictor, Input, Path
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

MODEL_CACHE = "diffusers-cache"

class Predictor(BasePredictor):
    '''Predictor class for StableDiffusion-v1'''

    def setup(self):
        '''
        Load the model into memory to make running multiple predictions efficient
        '''
        print("Loading pipeline...")

        self.inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                            "runwayml/stable-diffusion-inpainting",
                            cache_dir=MODEL_CACHE,
                            local_files_only=True,
                            ).to("cuda")
        
        self.inpaint_pipe.enable_xformers_memory_efficient_attention()

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        prompt: str = Input(
            description="Prompt to inpaint on",
            default="A room with a rosjf sofa"
        ),
        width: int = Input(
            description="Output image width; max 1024x768 or 768x1024 due to memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=512,
        ),
        image: Path = Input(
            description="Image to inpaint on.",
            default=None,
        ),
        mask: Path = Input(
            description="Inpainting mask",
            default=None,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=20
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        scheduler: str = Input(
            default="K-LMS",
            choices=["DDIM", "DDPM", "DPM-M", "DPM-S", "EULER-A", "EULER-D",
                     "HEUN", "IPNDM", "KDPM2-A", "KDPM2-D", "PNDM",  "K-LMS"],
            description="Choose a scheduler. If you use an init image, PNDM will be used",
        ),
        use_lora: bool = Input(
            default=False,
            description="Use a lora to inpaint the image")

    ) -> List[Path]:
        '''
        Run a single prediction on the model
        '''        

        seed = int.from_bytes(os.urandom(2), "big")

        if not image:
            raise ValueError("No image provided")

        image = Image.open(image).convert("RGB")
        mask = Image.open(mask).convert("RGB")

        self.inpaint_pipe.scheduler = make_scheduler(scheduler, self.inpaint_pipe.scheduler.config)
        generator = torch.Generator("cuda").manual_seed(seed)

        height = int(width * image.height / image.width)
        height = height - (height % 8)
        
        image = image.resize((width, height), Image.BILINEAR)
        mask = mask.resize((width, height), Image.BILINEAR)

        output_dirs = []
        
        timestart = time.time()

        if (use_lora):
            apply_lora(self.inpaint_pipe, f"rosjf-05.safetensors", weight=1.3)
        
        print("Time to load lora: ", time.time() - timestart)

        timestart = time.time()
        
        output = self.inpaint_pipe(prompt, 
                    image, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, mask_image=mask, 
                    width=width, height=height).images[0]

        print("Time to run inference: ", time.time() - timestart)
        
        output_path = f"tmp/output.png"
        output.save(output_path, optimize=True, quality=30)
        output_dirs.append(output_path)

        return output_dirs


def make_scheduler(name, config):
    '''
    Returns a scheduler from a name and config.
    '''
    return {
        "DDIM": DDIMScheduler.from_config(config),
        "DDPM": DDPMScheduler.from_config(config),
        "DPM-M": DPMSolverMultistepScheduler.from_config(config),
        "DPM-S": DPMSolverSinglestepScheduler.from_config(config),
        "EULER-A": EulerAncestralDiscreteScheduler.from_config(config),
        "EULER-D": EulerDiscreteScheduler.from_config(config),
        "HEUN": HeunDiscreteScheduler.from_config(config),
        "IPNDM": IPNDMScheduler.from_config(config),
        "KDPM2-A": KDPM2AncestralDiscreteScheduler.from_config(config),
        "KDPM2-D": KDPM2DiscreteScheduler.from_config(config),
        "PNDM": PNDMScheduler.from_config(config),
        "K-LMS": LMSDiscreteScheduler.from_config(config)
    }[name]
