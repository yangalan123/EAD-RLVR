import numpy as np
import torch
from typing import Union


def annealed_sampling_processor(token_ids: Union[list[int], tuple[int]], logits: torch.Tensor, 
                               exploration_temp: float = 1.0, stability_temp: float = 0.1, 
                               decay_freq: int = 50, global_step: int = 0,
                               decay_mode: str = 'both', warmup_period: int = 10,
                               decay_freq_increase_factor: int = 5) -> torch.Tensor:
    """
    Annealed sampling logits processor for vLLM.
    
    Args:
        token_ids: List of token IDs generated so far
        logits: Logits tensor from the model
        exploration_temp: Exploration temperature (higher = more exploration)
        stability_temp: Stability temperature (lower = more focused)
        decay_freq: Decay frequency for temperature annealing
        global_step: Current global optimization step
        decay_mode: Which annealing mode to use. Options: 'global_step', 'token_length', 'both', 'none'.
        warmup_period: If len(token_ids) < warmup_period, do not apply temperature scaling (default: 10)
        decay_freq_increase_factor: Factor by which decay_freq increases with global_step (default: 5)
    Returns:
        Modified logits tensor
    """
    # If in warmup period, do not apply temperature scaling
    if len(token_ids) < warmup_period:
        return logits
    
    # Calculate the current temperature based on the selected decay mode
    if decay_mode == 'global_step':
        current_temp = stability_temp + (exploration_temp - stability_temp) * np.exp(-global_step / decay_freq)
    elif decay_mode == 'token_length':
        current_temp = stability_temp + (exploration_temp - stability_temp) * np.exp(-len(token_ids) / (20 * decay_freq))
    elif decay_mode == 'both':
        _exploration_temp = exploration_temp * np.exp(-global_step / decay_freq)
        current_temp = stability_temp + (_exploration_temp - stability_temp) * np.exp(-len(token_ids) / (20 * decay_freq))
    elif decay_mode == "both_v_1_5":
        _decay_freq = min(decay_freq + decay_freq_increase_factor * global_step, 2000)
        current_temp = stability_temp + (exploration_temp - stability_temp) * np.exp(-len(token_ids) / (20 * _decay_freq))
    elif decay_mode == "negexp":
        # as we use -exp(x/d), we need to use a larger decay_freq to get a smaller temperature and to keep the temperature >= 0
        _decay_freq = min(decay_freq + decay_freq_increase_factor * global_step, 40000)
        current_temp = 1 + exploration_temp - np.exp(len(token_ids) / (20 * _decay_freq))
        # avoid temperature < stability_temp
        current_temp = max(current_temp, stability_temp)
    elif decay_mode == "steps_variant":
        # use the same temperature at all positions, no matter how long token_ids is
        # the temperature gradually increases from stability_temp to exploration_temp
        current_temp = exploration_temp + (stability_temp - exploration_temp) * np.exp(-global_step / decay_freq)
        current_temp = min(current_temp, 1.0)
    elif decay_mode == "steps_variant_rev":
        # use the same temperature at all positions, no matter how long token_ids is
        # the temperature gradually increases from stability_temp to exploration_temp
        current_temp = stability_temp + (exploration_temp - stability_temp) * np.exp(-global_step / decay_freq)
        current_temp = max(current_temp, 0.1)
    elif decay_mode == 'none':
        current_temp = exploration_temp
    else:
        raise ValueError(f"Unknown decay_mode: {decay_mode}")

    # Apply temperature scaling to logits
    logits = logits / current_temp
    
    return logits