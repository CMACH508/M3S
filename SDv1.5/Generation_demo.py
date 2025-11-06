import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import numpy as np
import torch
#torch.cuda.set_device(0)
from pathlib import Path
# from PIL import Image
from IPython.display import display
from config import RunConfig
from run import run
from notebooks.prompts_dict import sketchy_prompts, common_prompts
# prompts=common_prompts
prompts=common_prompts

skip_steps = 10
CFG = 15
swap_guidance_scale = 15
for j in [1,2,3,4,5]:
  style_image_path = f"./style/style{j}/"
  image_path_list = sorted(os.listdir(style_image_path))*3
  image_path_list = image_path_list[0:len(prompts)]
  seed = 42 
  sparse_weight = 0
  interpolation = 0.1
  for i, image_path in enumerate(image_path_list):
      print(prompts[i+1])
      full_image_path = os.path.join(style_image_path, image_path)
      domain_name = prompts[i+1]
      config = RunConfig(
          skip_steps=skip_steps,
          app_image_path=Path(full_image_path),
          struct_image_path=Path(full_image_path),
          output_path=Path(f'neurips_demo/CFG{CFG}_sfg{swap_guidance_scale}_skip{skip_steps}/style{j}_{sparse_weight}_int_{interpolation}'),
          #output_path=Path(f'kv_swap/style{j}/'),
          domain_name=domain_name,
          seed=seed,
          swap_guidance_scale=swap_guidance_scale,
          CFG=CFG,
          mix_style=False,
          sparse_weight=sparse_weight,
          load_latents=False,
          interpolation=interpolation,
          resize=False
      )
      images = run(cfg=config)
      torch.cuda.empty_cache()



skip_steps = 10
CFG = 15
swap_guidance_scale = 25
for j in [6]:
  style_image_path = f"./style/style{j}/"
  image_path_list = sorted(os.listdir(style_image_path))*3
  image_path_list = image_path_list[0:len(prompts)]
  seed = 42 
  sparse_weight = 60
  interpolation = 0.05
  for i, image_path in enumerate(image_path_list):
      print(prompts[i+1])
      full_image_path = os.path.join(style_image_path, image_path)
      domain_name = prompts[i+1]
      config = RunConfig(
          skip_steps=skip_steps,
          app_image_path=Path(full_image_path),
          struct_image_path=Path(full_image_path),
          output_path=Path(f'neurips_demo/CFG{CFG}_sfg{swap_guidance_scale}_skip{skip_steps}/style{j}_{sparse_weight}_int_{interpolation}'),
          #output_path=Path(f'kv_swap/style{j}/'),
          domain_name=domain_name,
          seed=seed,
          swap_guidance_scale=swap_guidance_scale,
          CFG=CFG,
          mix_style=False,
          sparse_weight=sparse_weight,
          load_latents=False,
          interpolation=interpolation,
          resize=False
      )
      images = run(cfg=config)
      torch.cuda.empty_cache()
