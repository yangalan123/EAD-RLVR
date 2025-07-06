import numpy as np
import torch
from typing import Union, Dict, Optional
import pickle
import os
from collections import defaultdict


class HistoricalDataManager:
    """
    Manages historical data for adaptive annealing based on previous rollout performance.
    
    Stores:
    - Last-round rollout token lengths L(x) for each prompt
    - Last-round advantages for each trajectory  
    - Current round log probabilities (for potential future use)
    """
    
    def __init__(self, cache_dir: Optional[str] = None, max_cache_size: int = 10000):
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size
        self.history_cache: Dict[str, Dict] = defaultdict(dict)
        
        # Load existing cache if available
        if cache_dir and os.path.exists(cache_dir):
            self._load_cache()
    
    def _get_cache_file(self) -> str:
        """Get the cache file path."""
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, "annealing_history.pkl")
    
    def _load_cache(self):
        """Load historical data from disk."""
        cache_file = self._get_cache_file()
        if cache_file and os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    self.history_cache = pickle.load(f)
                print(f"Loaded {len(self.history_cache)} historical entries from {cache_file}")
            except Exception as e:
                print(f"Warning: Could not load historical cache: {e}")
    
    def _save_cache(self):
        """Save historical data to disk."""
        cache_file = self._get_cache_file()
        if cache_file:
            try:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.history_cache, f)
            except Exception as e:
                print(f"Warning: Could not save historical cache: {e}")
    
    def update_history(self, uids: np.ndarray, token_lengths: np.ndarray, 
                      advantages: Optional[np.ndarray] = None, 
                      log_probs: Optional[np.ndarray] = None):
        """
        Update historical data with new rollout information.
        
        Args:
            uids: Array of unique identifiers for each trajectory
            token_lengths: Array of response token lengths for each trajectory
            advantages: Optional array of advantages for each trajectory
            log_probs: Optional array of log probabilities for each trajectory
        """
        for i, uid in enumerate(uids):
            uid_str = str(uid)
            
            # Store token length (this is the key metric for annealing)
            self.history_cache[uid_str]['last_token_length'] = int(token_lengths[i])
            
            # Store additional data if provided
            if advantages is not None:
                self.history_cache[uid_str]['last_advantage'] = float(advantages[i])
            if log_probs is not None:
                # Store mean log prob for the sequence
                self.history_cache[uid_str]['last_log_prob'] = float(log_probs[i])
            
            # Add timestamp for potential cleanup
            self.history_cache[uid_str]['last_updated'] = len(self.history_cache)
        
        # Cleanup old entries if cache is too large
        if len(self.history_cache) > self.max_cache_size:
            self._cleanup_cache()
        
        # Save to disk periodically
        if len(self.history_cache) % 1000 == 0:
            self._save_cache()
    
    def get_last_token_length(self, uid: str, default_length: int = 50) -> int:
        """
        Get the last rollout token length for a given UID.
        
        Args:
            uid: Unique identifier for the trajectory
            default_length: Default length to return if no history exists
            
        Returns:
            The last token length, or default_length if no history exists
        """
        uid_str = str(uid)
        if uid_str in self.history_cache:
            return self.history_cache[uid_str].get('last_token_length', default_length)
        return default_length
    
    def get_last_advantage(self, uid: str) -> Optional[float]:
        """Get the last advantage for a given UID."""
        uid_str = str(uid)
        if uid_str in self.history_cache:
            return self.history_cache[uid_str].get('last_advantage', None)
        return None
    
    def _cleanup_cache(self):
        """Remove oldest entries to keep cache size manageable."""
        if len(self.history_cache) <= self.max_cache_size:
            return
        
        # Sort by last_updated and keep only the most recent entries
        sorted_items = sorted(self.history_cache.items(), 
                            key=lambda x: x[1].get('last_updated', 0))
        
        # Keep the most recent entries
        items_to_keep = sorted_items[-self.max_cache_size:]
        self.history_cache = dict(items_to_keep)


# Global instance for easy access
_historical_manager = None

def get_historical_manager(cache_dir: Optional[str] = None) -> HistoricalDataManager:
    """Get or create the global historical data manager."""
    global _historical_manager
    if _historical_manager is None:
        _historical_manager = HistoricalDataManager(cache_dir=cache_dir)
    return _historical_manager


def annealed_sampling_processor(token_ids: Union[list[int], tuple[int]], logits: torch.Tensor, 
                               exploration_temp: float = 1.0, stability_temp: float = 0.1, 
                               decay_freq: int = 50, global_step: int = 0,
                               decay_mode: str = 'both', warmup_period: int = 10,
                               adaptive_decay: bool = False, uid: Optional[str] = None,
                               historical_manager: Optional[HistoricalDataManager] = None,
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
        decay_mode: Which annealing mode to use. Options: 'global_step', 'token_length', 'both', 'none', 'adaptive'.
        warmup_period: If len(token_ids) < warmup_period, do not apply temperature scaling (default: 10)
        adaptive_decay: Whether to use adaptive decay based on historical performance
        uid: Unique identifier for the current trajectory (required for adaptive decay)
        historical_manager: Historical data manager instance (optional, will use global if None)
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
    elif decay_mode == 'adaptive':
        # New adaptive decay mode based on historical performance
        if adaptive_decay and uid is not None:
            # Get historical manager
            if historical_manager is None:
                historical_manager = get_historical_manager()
            
            # Get last token length for this UID
            last_token_length = historical_manager.get_last_token_length(uid, default_length=50)
            
            # Use adaptive decay: exp(-len(token_ids) / (0.1 * L(x)))
            adaptive_decay_rate = len(token_ids) / (0.1 * last_token_length)
            current_temp = stability_temp + (exploration_temp - stability_temp) * np.exp(-adaptive_decay_rate)
        else:
            # Fallback to standard token_length mode
            current_temp = stability_temp + (exploration_temp - stability_temp) * np.exp(-len(token_ids) / (20 * decay_freq))
    elif decay_mode == 'none':
        current_temp = exploration_temp
    else:
        raise ValueError(f"Unknown decay_mode: {decay_mode}")

    # Apply temperature scaling to logits
    logits = logits / current_temp
    
    return logits