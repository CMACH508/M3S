from typing import Any, Callable, Dict, List, Optional, Tuple, Union
#from diffusers.image_processor import PipelineImageInput
import numpy as np
import torch
import torchvision
from diffusers import StableDiffusionXLPipeline
from diffusers.models import AutoencoderKL
#from diffusers.pipelines.stable_diffusion import StableDiffusionXLPipelineOutput

from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionSafetyChecker
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg #bug
from diffusers.schedulers import KarrasDiffusionSchedulers
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor,CLIPTextModelWithProjection,CLIPVisionModelWithProjection
from torchvision import transforms
from config import Range
from diffusers import UNet2DConditionModel
import torch.nn.functional as F
import torch
from PIL import Image
#from diffusers.callbacks import PipelineCallback, MultiPipelineCallbacks

torch.autograd.set_detect_anomaly(True)
device = "cuda" if torch.cuda.is_available() else "cpu"


class CrossImageAttentionStableDiffusionPipeline(StableDiffusionXLPipeline):
    """ A modification of the standard StableDiffusionPipeline to incorporate our cross-image attention."""

    def __init__(self, vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,#FreeUUNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        force_zeros_for_empty_prompt: bool = True,
        ):
        super().__init__(
            vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2, unet, scheduler,force_zeros_for_empty_prompt# safety_checker
        )

    @torch.no_grad()
    def __call__(
            self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image=None,#: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end=None,#: Optional[
        #     Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        # ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            swap_guidance_scale: float = 1.0,
            cross_image_attention_range: Range = Range(10, 90),
            # DDPM addition
            zs: Optional[List[torch.Tensor]] = None,
            sparse_weight=3e-5,
            clip_weight=1e-2,
            run_config=None,
            **kwargs,
    ):
        callback = kwargs.pop("callback", None)
        callback_steps = 1#kwargs.pop("callback_steps", None)
        self.run_config = run_config

        # 0. Default height and width to unet
        height = 128#height or self.unet.config.sample_size * self.vae_scale_factor
        width = 128#width or self.unet.config.sample_size * self.vae_scale_factor
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        # self.check_inputs(
        #     prompt,
        #     prompt_2,
        #     height,
        #     width,
        #     callback_steps,
        #     negative_prompt,
        #     negative_prompt_2,
        #     prompt_embeds,
        #     negative_prompt_embeds,
        #     pooled_prompt_embeds,
        #     negative_pooled_prompt_embeds,
        #     ip_adapter_image,
        #     ip_adapter_image_embeds,
        #     callback_on_step_end_tensor_inputs,
        # )
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._interrupt = False


        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            #do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            #clip_skip=self.clip_skip,
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

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            #text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        op = tqdm(timesteps[-zs[0].shape[0]:])
        n_timesteps = len(timesteps[-zs[0].shape[0]:])

        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        count = 0
        for t in op:
            i = t_to_idx[int(t)]

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            added_cond_kwargs = {"text_embeds": add_text_embeds.to(torch.bfloat16)[:4], "time_ids": add_time_ids.to(torch.bfloat16)[:4]}
            if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                added_cond_kwargs["image_embeds"] = image_embeds

            noise_pred_swap = self.unet(
                latent_model_input[0:4].to(torch.bfloat16),
                t,
                encoder_hidden_states=prompt_embeds[0:4],
                added_cond_kwargs=added_cond_kwargs,
                timestep_cond=timestep_cond,
                cross_attention_kwargs={'perform_swap': True},
                return_dict=False,
            )[0]
            noise_pred_swap = torch.cat([noise_pred_swap, noise_pred_swap[1:3]], dim=0)

            tmp_added_cond_kwargs = {"text_embeds": torch.cat([add_text_embeds.to(torch.bfloat16)[:1],add_text_embeds.to(torch.bfloat16)[3:4]],dim=0),
                                    "time_ids": torch.cat([add_time_ids.to(torch.bfloat16)[:1],add_time_ids.to(torch.bfloat16)[3:4]],dim=0),}
            noise_pred_no_swap = self.unet(
                torch.cat([latent_model_input[0:1], latent_model_input[3:4]], dim=0).to(torch.bfloat16), #latent_model_input,
                t,
                encoder_hidden_states=torch.cat([prompt_embeds[0:1],prompt_embeds[3:4]], dim=0), #prompt_embeds,
                cross_attention_kwargs={'perform_swap': False},
                added_cond_kwargs=tmp_added_cond_kwargs,
                timestep_cond=timestep_cond,
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
                                                     max(swap_guidance_scale /3, 1.0),
                                                     n_timesteps)
                CFG_strengths = np.linspace(guidance_scale,
                                                     max(guidance_scale/ 1, 5),
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
                                                     max(swap_guidance_scale / 3.0, 1.0),
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
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
            elif latents.dtype != self.vae.dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    self.vae = self.vae.to(latents.dtype)

            # unscale/denormalize the latents
            # denormalize with the mean and std if available and not None
            has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
            else:
                latents = latents / self.vae.config.scaling_factor

            image = self.vae.decode(latents.to(torch.bfloat16), return_dict=False)[0]
            image[(image.mean(dim=1,keepdim=True) > 0.7).repeat(1,3,1,1)] = 1
            image = image.mean(dim=1,keepdim=True).repeat(1,3,1,1)


            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

        # # Offload all models
        # self.maybe_free_model_hooks()

        has_nsfw_concept = None
        # if has_nsfw_concept is None:
        #     do_denormalize = [True] * image.shape[0]
        # else:
        #     do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        # image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)


        if not return_dict:
            return (image, has_nsfw_concept)
        
        #return StableDiffusionXLPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
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
            #down_latent = torchvision.transforms.Resize((64, 64),antialias=False)(latent)
            down_latent = torchvision.transforms.RandomCrop((64, 64))(latent)
            #latent.grad = None
            
            image = self.vae.decode(down_latent.to(torch.bfloat16) / self.vae.config.scaling_factor, return_dict=False)[0]
            grad_x, grad_y, edge_map = self.compute_gradient(image)
            # sparse_loss = - (torch.abs(grad_x) + torch.abs(grad_y)).mean()
            edge_map = (1-edge_map)*2-1
            sparse_loss = torch.abs(1-2*image+2).mean() - (torch.abs(grad_x) + torch.abs(grad_y)).mean()#+tv_loss_second_order(image)
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
        
        sobel_x = torch.tensor(Gx, dtype=torch.bfloat16).view(1, 1, 3, 3)
        sobel_y = torch.tensor(Gy, dtype=torch.bfloat16).view(1, 1, 3, 3)
            
        sobel_x = sobel_x.to(image.device).to(torch.bfloat16)
        sobel_y = sobel_y.to(image.device).to(torch.bfloat16)
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

