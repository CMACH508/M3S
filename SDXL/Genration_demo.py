import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import numpy as np
import torch
from pathlib import Path
from IPython.display import display
from config import RunConfig
from run import run
from notebooks.prompts_dict import prompts

skip_steps = 0
CFG = 18
swap_guidance_scale = 12
for j in [1,2,3,4,5]:
  style_image_path = f"./style/style{j}/"
  image_path_list = sorted(os.listdir(style_image_path))*3
  image_path_list = image_path_list[0:len(prompts)]
  seed = 42 
  sparse_weight = 0
  interpolation =0.1
  for i, image_path in enumerate(image_path_list):
      print(prompts[i+1])
      full_image_path = os.path.join(style_image_path, image_path)
      domain_name = prompts[i+1]
      config = RunConfig(
          skip_steps=skip_steps,
          app_image_path=Path(full_image_path),
          struct_image_path=Path(full_image_path),
          output_path=Path(f'neurips_demo_1024/CFG{CFG}_sfg{swap_guidance_scale}_skip{skip_steps}/style{j}_{sparse_weight}_int_{interpolation}'),
          domain_name=domain_name,
          seed=seed,
          swap_guidance_scale=swap_guidance_scale,
          CFG=CFG,
          mix_style=False,
          sparse_weight=sparse_weight,
          load_latents=False,
          interpolation=interpolation,
          Inject_layer=[1,9,17,25,33,
                        41,49,57,69,71]
      )
      images = run(cfg=config)
      torch.cuda.empty_cache()


skip_steps = 0
CFG = 7.5
swap_guidance_scale = 20
for j in [6]:
  style_image_path = f"./style/style{j}/"
  image_path_list = sorted(os.listdir(style_image_path))*3
  image_path_list = image_path_list[0:len(prompts)]
  seed = 42 
  sparse_weight = 0
  interpolation =0.05
  for i, image_path in enumerate(image_path_list):
      print(prompts[i+1])
      full_image_path = os.path.join(style_image_path, image_path)
      domain_name = prompts[i+1]
      config = RunConfig(
          skip_steps=skip_steps,
          app_image_path=Path(full_image_path),
          struct_image_path=Path(full_image_path),
          output_path=Path(f'neurips_demo_1024/CFG{CFG}_sfg{swap_guidance_scale}_skip{skip_steps}/style{j}_{sparse_weight}_int_{interpolation}'),
          domain_name=domain_name,
          seed=seed,
          swap_guidance_scale=swap_guidance_scale,
          CFG=CFG,
          mix_style=False,
          sparse_weight=sparse_weight,
          load_latents=False,
          interpolation=interpolation,
          Inject_layer=[1,9,17,25,33,
                        41,49,57,69,71]
      )
      images = run(cfg=config)
      torch.cuda.empty_cache()
