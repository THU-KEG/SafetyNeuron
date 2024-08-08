import re
import random
import os
import torch
import pandas as pd
import numpy as np
from typing import Optional, Union
from safetensors import safe_open


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_act_name(
    name: str,
    layer: Optional[Union[int, str]] = None,
    layer_type: Optional[str] = None,
):
    """
    Helper function to convert shorthand to an activation name. Pretty hacky, intended to be useful for short feedback
    loop hacking stuff together, more so than writing good, readable code. But it is deterministic!

    Returns a name corresponding to an activation point in a TransformerLens model.

    Args:
         name (str): Takes in the name of the activation. This can be used to specify any activation name by itself.
         The code assumes the first sequence of digits passed to it (if any) is the layer number, and anything after
         that is the layer type.

         Given only a word and number, it leaves layer_type as is.
         Given only a word, it leaves layer and layer_type as is.

         Examples:
             get_act_name('embed') = get_act_name('embed', None, None)
             get_act_name('k6') = get_act_name('k', 6, None)
             get_act_name('scale4ln1') = get_act_name('scale', 4, 'ln1')

         layer (int, optional): Takes in the layer number. Used for activations that appear in every block.

         layer_type (string, optional): Used to distinguish between activations that appear multiple times in one block.

    Full Examples:

    get_act_name('k', 6, 'a')=='blocks.6.attn.hook_k'
    get_act_name('pre', 2)=='blocks.2.mlp.hook_pre'
    get_act_name('embed')=='hook_embed'
    get_act_name('normalized', 27, 'ln2')=='blocks.27.ln2.hook_normalized'
    get_act_name('k6')=='blocks.6.attn.hook_k'
    get_act_name('scale4ln1')=='blocks.4.ln1.hook_scale'
    get_act_name('pre5')=='blocks.5.mlp.hook_pre'
    """
    if (
        ("." in name or name.startswith("hook_"))
        and layer is None
        and layer_type is None
    ):
        # If this was called on a full name, just return it
        return name
    match = re.match(r"([a-z]+)(\d+)([a-z]?.*)", name)
    if match is not None:
        name, layer, layer_type = match.groups(0)

    layer_type_alias = {
        "a": "self_attn",
        "m": "mlp",
        "b": "",
        "l": "",
        "block": "",
        "blocks": "",
        "layer": "",
        "layers": "",
        "attention": "self_attn",
    }

    act_name_alias = {
        "attn": "pattern",
        "attn_logits": "attn_scores",
        "key": "k",
        "query": "q",
        "value": "v",
        "mlp_pre": "pre",
        "mlp_mid": "mid",
        "mlp_post": "post",
    }

    layer_norm_names = ["scale", "normalized"]

    if name in act_name_alias:
        name = act_name_alias[name]

    full_act_name = ""
    if layer is not None:
        full_act_name += f"model.layers.{layer}."
    if name in [
        "k",
        "v",
        "q",
        "z",
        "rot_k",
        "rot_q",
        "result",
        "pattern",
        "attn_scores",
    ]:
        layer_type = "self_attn"
    elif name in ["pre", "post", "mid", "pre_linear"]:
        layer_type = "mlp"
    elif layer_type in layer_type_alias:
        layer_type = layer_type_alias[layer_type]

    if layer_type:
        full_act_name += f"{layer_type}."
    full_act_name += f"hook_{name}"

    if name in layer_norm_names and layer is None:
        full_act_name = f"norm.{full_act_name}"
    return full_act_name

def load_adapter_weights(path):
    adapter_weights = {}
    try:
        with safe_open(os.path.join(path, "adapter_model.safetensors"), framework="pt", device="cpu") as f:
            for key in f.keys():
                adapter_weights[key] = f.get_tensor(key)
    except FileNotFoundError:
        adapter_weights = torch.load(os.path.join(path, "adapter_model.bin"))
    return adapter_weights

def topk_index(tensor, k):
    if k < 0:
        k = tensor.numel()
    index = tensor.flatten().topk(k=k)[1]
    shape = tensor.shape
    out = []
    for dim in reversed(shape):
        out.append((index % dim).tolist())
        index = index // dim
    return zip(*tuple(reversed(out)))