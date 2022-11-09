import argparse
import json
import os

import torch
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
# from colossalai.logging import get_dist_logger
from torch.distributed import all_reduce, get_world_size, get_rank, is_initialized

CONFIG = None
ARGS = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--data-path', type=str)
    parser.add_argument('--vocab-file', type=str)
    parser.add_argument('--log-path', type=str)
    parser.add_argument('--use-mpi', action="store_true")
    parser.add_argument('--do-eval', action="store_true")
    return parser.parse_args()


def get_args():
    global ARGS
    if ARGS is None:
        ARGS = parse_args()
    return ARGS


def load_config():
    args = get_args()
    config_file = args.config

    assert os.path.exists(config_file), 'No valid config file found.'

    config = dict()
    with open(config_file, 'r') as f:
        cfg = json.load(f)
        for k, v in cfg.items():
            config[k] = v

    return config


def get_config():
    global CONFIG
    if CONFIG is None:
        CONFIG = load_config()
    return CONFIG


def print_log(msg, *args, **kwargs):
    # get_dist_logger().info(msg, *args, ranks=[0], **kwargs)
    rank = get_rank() if is_initialized() else 0
    if rank == 0:
        print("\n" + msg)


class ModelFromHF(torch.nn.Module):

    def __init__(self, config, model_cls):
        super().__init__()
        self.module = model_cls(config)
        if config.checkpoint:
            self.module.apply(self.set_checkpointing)

    def set_checkpointing(self, module):
        if hasattr(module, 'gradient_checkpointing'):
            module.gradient_checkpointing = True

    def forward(self, *args, **kwargs):
        output = self.module(*args, **kwargs)
        return output.logits


def get_tflops(iter_time):
    config = get_config()
    # flops = numel * num_tokens * 2 / get_world_size()
    model_cfg = config['model']
    vocab_size = model_cfg['vocab_size']
    hidden_size = model_cfg['hidden_size']
    num_layers = model_cfg['depth']
    seq_length = model_cfg['seq_length']
    global_batch_size = config['hyperparameter'].get('global_batch_size', config['hyperparameter']['batch_size'])
    flops = 4 * global_batch_size * seq_length * num_layers * hidden_size * (6 * hidden_size + seq_length)
    if config['model'].get('checkpoint', False):
        flops *= 4
    else:
        flops *= 3
    flops += 6 * global_batch_size * seq_length * hidden_size * vocab_size
    return (flops / 1e12) / (iter_time * get_world_size())


def get_model_size(model: torch.nn.Module):
    numel = torch.tensor(sum(p.numel() for p in model.parameters())).to(torch.cuda.current_device())
    all_reduce(numel, group=gpc.get_group(ParallelMode.MODEL))
    return numel.item()


def get_gpu_memory_mb():
    return torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1024**2
