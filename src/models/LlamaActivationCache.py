
from typing import Dict
import torch

from src.utils import get_act_name
from transformer_lens.ActivationCache import ActivationCache


class LlamaActivationCache(ActivationCache):
    def __init__(self, cache_dict: Dict[str, torch.Tensor], model, has_batch_dim: bool = True):
        super().__init__(cache_dict, model, has_batch_dim)

    def __repr__(self) -> str:
        """Representation of the ActivationCache.

        Special method that returns a string representation of an object. It's normally used to give
        a string that can be used to recreate the object, but here we just return a string that
        describes the object.
        """
        return f"LlamaActivationCache with keys {list(self.cache_dict.keys())}"

    def __getitem__(self, key) -> torch.Tensor:
        """Retrieve Cached Activations by Key or Shorthand.

        Enables direct access to cached activations via dictionary-style indexing using keys or
        shorthand naming conventions. It also supports tuples for advanced indexing, with the
        dimension order as (get_act_name, layer_index, layer_type).

        Args:
            key:
                The key or shorthand name for the activation to retrieve.

        Returns:
            The cached activation tensor corresponding to the given key.
        """
        if key in self.cache_dict:
            return self.cache_dict[key]
        elif type(key) == str:
            return self.cache_dict[get_act_name(key)]
        else:
            if len(key) > 1 and key[1] is not None:
                if key[1] < 0:
                    # Supports negative indexing on the layer dimension
                    key = (key[0], self.model.model.config.num_hidden_layers + key[1], *key[2:])
            return self.cache_dict[get_act_name(*key)]