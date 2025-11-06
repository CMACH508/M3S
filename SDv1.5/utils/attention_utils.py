import math
import torch
from constants import OUT_INDEX, STRUCT_INDEX, STYLE_INDEX


def should_mix_keys_and_values(model, hidden_states: torch.Tensor) -> bool:
    """ Verify whether we should perform the mixing in the current timestep. """
    is_in_32_timestep_range = (
            model.config.cross_attn_32_range.start <= model.step < model.config.cross_attn_32_range.end
    )
    is_in_64_timestep_range = (
            model.config.cross_attn_64_range.start <= model.step < model.config.cross_attn_64_range.end
    )
    is_hidden_states_32_square = (hidden_states.shape[1] == 32 ** 2)
    is_hidden_states_64_square = (hidden_states.shape[1] == 64 ** 2)
    should_mix = (is_in_32_timestep_range and is_hidden_states_32_square) or \
                 (is_in_64_timestep_range and is_hidden_states_64_square)
    return should_mix


def compute_scaled_dot_product_attention(Q, K, V, edit_map=False, is_cross=False, contrast_strength=1.0, attn_control=None):
    """ Compute the scale dot product attention, potentially with our contrasting operation. """
    score = (Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1)))
    if attn_control is not None:
        score[STYLE_INDEX,:,:,attn_control:] = -10000
        score[STRUCT_INDEX,:,:,attn_control:] = -10000


    attn_weight = torch.softmax(score, dim=-1)
    if edit_map and not is_cross:
        attn_weight[OUT_INDEX] = torch.stack([
            enhance_tensor(attn_weight[OUT_INDEX][head_idx], contrast_factor=contrast_strength)
            for head_idx in range(attn_weight.shape[1])
        ])
    return attn_weight @ V, attn_weight


def enhance_tensor(tensor: torch.Tensor, contrast_factor: float = 1.67) -> torch.Tensor:
    """ Compute the attention map contrasting. """
    bridge= tensor.mean(dim=-1,keepdim=True)
    adjusted_tensor = (tensor - bridge) * contrast_factor + bridge
    return adjusted_tensor
