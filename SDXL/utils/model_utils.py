import torch
from diffusers import DDIMScheduler

from models.stable_diffusion import CrossImageAttentionStableDiffusionPipeline
from models.unet_2d_condition import FreeUUNet2DConditionModel
from diffusers import UNet2DConditionModel

def get_stable_diffusion_model() -> CrossImageAttentionStableDiffusionPipeline:
    print("Loading Stable Diffusion model...")
    device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')
    pipe = CrossImageAttentionStableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                                      safety_checker=None,torch_dtype=torch.bfloat16, variant="fp16", use_safetensors=True).to(device)
    pipe.scheduler = DDIMScheduler.from_config("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
    print("Done.")
    return pipe
