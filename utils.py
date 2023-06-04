from safetensors.torch import load_file
from sdscripts.networks.lora import create_network_from_weights
import torch

def apply_lora(pipe, lora_path, weight:float = 1.0):
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    unet = pipe.unet

    sd = load_file(lora_path)
    lora_network, sd = create_network_from_weights(weight, None, vae, text_encoder, unet, sd)
    lora_network.apply_to(text_encoder, unet)
    lora_network.load_state_dict(sd)
    lora_network.to("cuda", dtype=torch.float16)
