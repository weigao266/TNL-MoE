import logging
import os
import sys

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F

from .norm import SimpleRMSNorm as SimpleRMSNormTorch
from .srmsnorm_triton import SimpleRMSNorm as SimpleRMSNormTriton

use_triton = eval(os.environ.get("use_triton", default="True"))
debug = eval(os.environ.get("debug", default="False"))

if use_triton:
    SimpleRMSNorm = SimpleRMSNormTriton
else:
    SimpleRMSNorm = SimpleRMSNormTorch

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("print_config")

BASE_DIM = 256


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def logging_info(string):
    if is_main_process():
        logger.info(string)


def print_params(**kwargs):
    if is_main_process():
        logger.info(f"start print config of {kwargs['__class__']}")
        for key in kwargs:
            if key in ["__class__", "self"]:
                continue
            logger.info(f"{key}: {kwargs[key]}")
        logger.info(f"end print config of {kwargs['__class__']}")


def print_config(config):
    if is_main_process():
        logger.info(f"start print config of {config['__class__']}")
        for key in config:
            if key in ["__class__", "self"]:
                continue
            logger.info(f"{key}: {config[key]}")
        logger.info(f"end print config of {config['__class__']}")


def print_module(module):
    named_modules = set()
    for p in module.named_modules():
        named_modules.update([p[0]])
    named_modules = list(named_modules)

    string_repr = ""
    for p in module.named_parameters():
        name = p[0].split(".")[0]
        if name not in named_modules:
            string_repr = (string_repr + "(" + name + "): " + "Tensor(" +
                           str(tuple(p[1].shape)) + ", requires_grad=" +
                           str(p[1].requires_grad) + ")\n")

    return string_repr.rstrip("\n")


def get_activation_fn(activation):
    if debug:
        logger.info(f"activation: {activation}")
    if activation == "gelu":
        return F.gelu
    elif activation == "relu":
        return F.relu
    elif activation == "elu":
        return F.elu
    elif activation == "sigmoid":
        return F.sigmoid
    elif activation == "exp":

        def f(x):
            with torch.no_grad():
                x_max = torch.max(x, dim=-1, keepdims=True).values
            y = torch.exp(x - x_max)

            return y

        return f
    elif activation == "leak":
        return F.leaky_relu
    elif activation == "1+elu":

        def f(x):
            return 1 + F.elu(x)

        return f
    elif activation == "2+elu":

        def f(x):
            return 2 + F.elu(x)

        return f
    elif activation == "silu" or activation == "swish":
        return F.silu
    elif activation == "sine":
        return torch.sin
    else:
        logger.info(
            f"activation: does not support {activation}, use Identity!!!")
        return lambda x: x


def get_norm_fn(norm_type):
    if norm_type == "simplermsnorm":
        return SimpleRMSNorm
    else:
        return nn.LayerNorm


def convert_to_multiple_of_base(x):
    return BASE_DIM * ((x + BASE_DIM - 1) // BASE_DIM)
