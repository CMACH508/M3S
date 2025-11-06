# app_image_path:  style 1
# struct_image_path: style 2
# CFG: ω_1
# swap_guidance_scale: ω_2
# domain_name: prompt (there is no need to add the prefix "a sketch of")
# alpha: style tendency η
# interpolation: Linear smoothing λ
# sparse_weight: sparse weight for abstract sketch

# single-style
CUDA_VISIBLE_DEVICES=0 python run.py \
--seed 42 \
--skip_steps 0 \
--app_image_path "style/style1/0.png" \
--struct_image_path "style/style1/0.png" \
--output_path ./results/single \
--mix_style False \
--domain_name 'a dog' \
--swap_guidance_scale 15 \
--CFG 15 \
--sparse_weight 0 \
--interpolation 0.1 \
--alpha 0.0 \

# multi-style
CUDA_VISIBLE_DEVICES=0 python run.py \
--seed 42 \
--skip_steps 0 \
--app_image_path "style/style5/3.png" \
--struct_image_path "style/style5/2.png" \
--output_path ./results/multi \
--mix_style True \
--domain_name 'an apple' \
--swap_guidance_scale 7.5 \
--CFG 7.5 \
--sparse_weight 0 \
--interpolation 0.1 \
--alpha 0.5 \

# abstract-sketch
CUDA_VISIBLE_DEVICES=0 python run.py \
--seed 42 \
--skip_steps 0 \
--app_image_path "style/style6/n01846331_5009-1.png" \
--struct_image_path "style/style6/n01846331_5009-1.png" \
--output_path ./results/single \
--mix_style False \
--domain_name 'a cat' \
--swap_guidance_scale 25 \
--CFG 15 \
--sparse_weight 60 \
--interpolation 0.05 \
--alpha 0.0 \
