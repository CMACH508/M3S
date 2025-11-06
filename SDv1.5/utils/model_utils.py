import torch
from diffusers import DDIMScheduler

from models.stable_diffusion import CrossImageAttentionStableDiffusionPipeline
from models.unet_2d_condition import FreeUUNet2DConditionModel

# Since 'runwayml/stable-diffusion-v1-5' is not available in diffusers, we need to load the model manually
# You can change the path below to point to your local model directory
# You can download the model from ModelScope

def get_stable_diffusion_model() -> CrossImageAttentionStableDiffusionPipeline:
    print("Loading Stable Diffusion model...")
    device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')
    pipe = CrossImageAttentionStableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                                      safety_checker=None).to(device)
    pipe.unet = FreeUUNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet").to(device)
    pipe.scheduler = DDIMScheduler.from_config("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    print("Done.")
    return pipe
