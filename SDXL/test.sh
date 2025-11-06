# app_image_path:  style 1
# struct_image_path: style 2
# CFG: ω_1
# swap_guidance_scale: ω_2
# domain_name: prompt (there is no need to add the prefix "a sketch of")
# alpha: style tendency η in the paper
# interpolation: Linear smoothing λ
# sparse_weight: sparse weight for abstract sketch
# Inject_layer: the layer to inject the style

python run.py \
--seed 42 \
--skip_steps 0 \
--app_image_path "style/style4/5.PNG" \
--struct_image_path "style/style4/5.PNG" \
--output_path ./results/single \
--mix_style False \
--domain_name 'a handsome' \
--swap_guidance_scale 12 \
--CFG 18 \
--sparse_weight 0 \
--interpolation 0.1 \
--alpha 0.0 \
--Inject_layer [1,9,17,25,33,41,49,57,69,71] \

python run.py \
--seed 2048 \
--skip_steps 0 \
--app_image_path "style/style5/15.png" \
--struct_image_path "style/style5/12.png" \
--output_path ./results/mix \
--mix_style True \
--domain_name 'a frog' \
--swap_guidance_scale 15 \
--CFG 12 \
--sparse_weight 0 \
--interpolation 0.0 \
--alpha 0.5 \
--Inject_layer [1,9,17,25,33,41,49,57,69,71] \