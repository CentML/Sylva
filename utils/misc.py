import numpy as np
import random
import torch

from transformers import set_seed


def print_rank_0(msg, rank=None):
    if rank is not None and rank <= 0:
        print(msg)
    elif torch.distributed.get_rank() == 0:
        print(msg)


def set_random_seed(seed):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(obj, device=None):
    if hasattr(obj, "items"):
        output = {}
        for k, v in obj.items():
            if device:
                output[k] = v.to(device)
            else:
                output[k] = v.cuda()
    else:
        if device:
            output = obj.to(device)
        else:
            output = obj.cuda()
    return output


def prepare(batch, model_name, device):
    batch = to_device(batch, device)
    if "gpt" in model_name:
        batch["label_smooth"] = 0.1
    return batch
