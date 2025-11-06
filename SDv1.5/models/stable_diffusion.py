from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from diffusers.models import AutoencoderKL
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionSafetyChecker
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg
from diffusers.schedulers import KarrasDiffusionSchedulers
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
from torchvision import transforms
from config import Range
from models.unet_2d_condition import FreeUUNet2DConditionModel
import torch.nn.functional as F
import torch
from PIL import Image
torch.autograd.set_detect_anomaly(True)
device = "cuda" if torch.cuda.is_available() else "cpu"


class CrossImageAttentionStableDiffusionPipeline(StableDiffusionPipeline):
    """ A modification of the standard StableDiffusionPipeline to incorporate our cross-image attention."""

    def __init__(self, vae: AutoencoderKL,
                 text_encoder: CLIPTextModel,
                 tokenizer: CLIPTokenizer,
                 unet: FreeUUNet2DConditionModel,#UNet2DConditionModel,
                 scheduler: KarrasDiffusionSchedulers,
                 safety_checker: StableDiffusionSafetyChecker,
                 feature_extractor: CLIPImageProcessor,
                 requires_safety_checker: bool = True):
        super().__init__(
            vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor, requires_safety_checker
        )

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            swap_guidance_scale: float = 1.0,
            cross_image_attention_range: Range = Range(10, 90),
            # DDPM addition
            zs: Optional[List[torch.Tensor]] = None,
            sparse_weight=3e-5,
            clip_weight=1e-2,
            run_config=None
    ):
        self.run_config = run_config
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        t_to_idx = {int(v): k for k, v in enumerate(timesteps[-zs[0].shape[0]:])}
        timesteps = timesteps[-zs[0].shape[0]:]

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        op = tqdm(timesteps[-zs[0].shape[0]:])
        n_timesteps = len(timesteps[-zs[0].shape[0]:])

        count = 0
        for t in op:
            i = t_to_idx[int(t)]

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred_swap = self.unet(
                latent_model_input[0:4],
                t,
                encoder_hidden_states=prompt_embeds[0:4],
                cross_attention_kwargs={'perform_swap': True},
                return_dict=False,
            )[0]
            noise_pred_swap = torch.cat([noise_pred_swap, noise_pred_swap[1:3]], dim=0)

            noise_pred_no_swap = self.unet(
                torch.cat([latent_model_input[0:1], latent_model_input[3:4]], dim=0), #latent_model_input,
                t,
                encoder_hidden_states=torch.cat([prompt_embeds[0:1],prompt_embeds[3:4]], dim=0), #prompt_embeds,
                cross_attention_kwargs={'perform_swap': False},
                return_dict=False,
            )[0]

            tmp = noise_pred_swap.clone()
            tmp[0] = noise_pred_no_swap[0]
            tmp[3] = noise_pred_no_swap[1]
            noise_pred_no_swap = tmp

            # perform guidance
            if do_classifier_free_guidance:
                noise_swap_pred_uncond, noise_swap_pred_text = noise_pred_swap.chunk(2)
                noise_no_swap_pred_uncond, noise_no_swap_pred_text= noise_pred_no_swap.chunk(2)
                swapping_strengths = np.linspace(swap_guidance_scale,
                                                     max(swap_guidance_scale /3, 0.0), 
                                                     n_timesteps)
                CFG_strengths = np.linspace(guidance_scale,
                                                     max(guidance_scale/ 1, 7.5),
                                                     n_timesteps)
                swapping_strength = swap_guidance_scale
                CFG_strength = CFG_strengths[count]
                if i>=0:
                    swapping_strength = swapping_strengths[count]

                noise_pred = noise_no_swap_pred_uncond + swapping_strength * (
                        noise_swap_pred_uncond - noise_no_swap_pred_uncond) + CFG_strength * (noise_swap_pred_text - noise_no_swap_pred_uncond)
            else:
                is_cross_image_step = cross_image_attention_range.start <= i <= cross_image_attention_range.end
                if swap_guidance_scale > 1.0 and is_cross_image_step:
                    swapping_strengths = np.linspace(swap_guidance_scale,
                                                     max(swap_guidance_scale / 8, 1.0),
                                                     n_timesteps)
                    swapping_strength = swapping_strengths[count]
                    noise_pred = noise_pred_no_swap + swapping_strength * (noise_pred_swap - noise_pred_no_swap)
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_swap, guidance_rescale=guidance_rescale)
                else:
                    noise_pred = noise_pred_swap

            eta_scheduler =[0,1,1]
            zs[0]=None
            latents = torch.stack([
                self.perform_ddpm_step(t_to_idx, zs[latent_idx], latents[latent_idx], t, 
                                        noise_pred[latent_idx], eta=eta_scheduler[latent_idx], count=count, prompt=prompt[latent_idx],
                                        sparse_weight=sparse_weight, clip_weight=clip_weight)
                for latent_idx in range(latents.shape[0])
            ])

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                # progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

            count += 1

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image[(image.mean(dim=1,keepdim=True) > 0.7).repeat(1,3,1,1)] = 1
            # image[(image.mean(dim=1,keepdim=True) < 0.5).repeat(1,3,1,1)] = -1
            image = image.mean(dim=1,keepdim=True).repeat(1,3,1,1)
            #image = (image.mean(dim=1,keepdim=True)>0.5).float()*2-1
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    def perform_ddpm_step(self, t_to_idx, zs, latents, t, noise_pred, eta, count=0, prompt=None,sparse_weight=0., clip_weight=0.):
        idx = t_to_idx[int(t)]
        z = zs[idx] if not zs is None else None
        # 1. get previous step value (=t-1)
        prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        # 2. compute alphas, betas
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
        if zs is None and sparse_weight!=0:
            if (40-self.run_config.skip_steps)<count:
                pred_original_sample = self.optimize_latent(pred_original_sample, prompt=prompt, sparse_weight=sparse_weight, CLIP_weight=clip_weight)
            # else:
            #     pred_original_sample = self.optimize_latent(pred_original_sample, prompt=prompt, sparse_weight=sparse_weight, CLIP_weight=clip_weight, flag=False)
        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        # variance = self.scheduler._get_variance(timestep, prev_timestep)
        variance = self.get_variance(t)
        std_dev_t = eta * variance ** (0.5)
        # Take care of asymetric reverse process (asyrp)
        model_output_direction = noise_pred
        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        # pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output_direction
        pred_sample_direction = (1 - alpha_prod_t_prev - eta * variance) ** (0.5) * model_output_direction
        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        # 8. Add noice if eta > 0
        if eta > 0:
            if z is None:
                z = torch.randn(noise_pred.shape, device=self.device)
            sigma_z = eta * variance ** (0.5) * z
            prev_sample = prev_sample + sigma_z
        return prev_sample

    def get_variance(self, timestep):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance

    def optimize_latent(self, latent, prompt="", sparse_weight=1e-4, CLIP_weight=1e-4, flag=True):
        tv_weight = sparse_weight
        sparse_grad, tv_grad = self.combined_loss(latent)
        latent = latent - sparse_weight*sparse_grad - tv_weight*tv_grad
        return latent

    
    def tv_loss(self, latent):
        with torch.enable_grad():
            latent = latent.unsqueeze(0)
            latent.requires_grad=True
            latent.grad = None 
            image = self.vae.decode(latent / self.vae.config.scaling_factor, return_dict=False)[0]
            tv_loss = tv_loss_second_order(image)
            tv_grad = torch.autograd.grad(tv_loss,latent)[0]
        return tv_grad.squeeze()
    
    def combined_loss(self, latent):
        with torch.enable_grad():
            latent = latent.unsqueeze(0).requires_grad_(True)
            latent.grad = None 
            image = self.vae.decode(latent / self.vae.config.scaling_factor, return_dict=False)[0]
            
            grad_x, grad_y, edge_map = self.compute_gradient(image)
            edge_map = (1-edge_map)*2-1
            sparse_loss = - (torch.abs(grad_x) + torch.abs(grad_y)).mean()#+torch.abs(1-2*image+2).mean() - (torch.abs(grad_x) + torch.abs(grad_y)).mean()#+tv_loss_second_order(image)
            #tv_loss = tv_loss_second_order(image)

            sparse_grad = torch.autograd.grad(
                outputs=sparse_loss,
                inputs=latent,
                create_graph=False,
                retain_graph=True  
            )[0].squeeze()
            sparse_grad = torch.clamp(sparse_grad, -0.001, 0.001)
            tv_grad=0
            
            return sparse_grad, tv_grad

    def compute_gradient(self, image):
        # Gx = [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]] 
        # Gy = [[-3, -10, -3], [0, 0, 0], [3, 10, 3]] 
        Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        
        sobel_x = torch.tensor(Gx, dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor(Gy, dtype=torch.float32).view(1, 1, 3, 3)
            
        sobel_x = sobel_x.to(image.device)
        sobel_y = sobel_y.to(image.device)
        image = image.mean(dim=1,keepdim=True)
        # sobel_x = sobel_x.repeat(image.shape[1], 1, 1, 1)
        # sobel_y = sobel_y.repeat(image.shape[1], 1, 1, 1)
        grad_x = F.conv2d(image, sobel_x, padding=1, groups=image.shape[1])  
        grad_y = F.conv2d(image, sobel_y, padding=1, groups=image.shape[1]) 
        edge_map = (grad_x.abs() + grad_y.abs()) / 8.0 
        return grad_x, grad_y, edge_map
    
    
def tv_loss_second_order(z):
    h_diff = z[:, :, 2:, :] + z[:, :, :-2, :] - 2 * z[:, :, 1:-1, :]
    w_diff = z[:, :, :, 2:] + z[:, :, :, :-2] - 2 * z[:, :, :, 1:-1]
    loss = torch.mean(h_diff.abs()) + torch.mean(w_diff.abs())
    return loss
