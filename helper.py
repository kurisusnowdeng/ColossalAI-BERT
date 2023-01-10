import math
import os

import colossalai
import colossalai.nn as col_nn
import numpy as np
import torch
from colossalai.amp import AMP_TYPE
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import FusedAdam, HybridAdam
from colossalai.utils import get_current_device, get_dataloader
from colossalai.zero.init_ctx import ZeroInitContext
from colossalai.zero.shard_utils import TensorShardStrategy
from datasets import load_from_disk
from torch.cuda import reset_peak_memory_stats
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertConfig, BertTokenizer, DataCollatorForLanguageModeling

from model import BertForMaskedLM, BertMaskedLMLoss, bias_dropout_add, bias_gelu
from utils import ModelFromHF, get_args, get_config, get_model_size, print_log

_bert_base = dict(
    seq_length=512,
    hidden_size=768,
    num_heads=12,
    depth=12,
    ff_size=3072,
)

_bert_large = dict(
    seq_length=512,
    hidden_size=1024,
    num_heads=16,
    depth=24,
    ff_size=4096,
)

_bert_oppo = dict(
    seq_length=512,
    hidden_size=2048,
    num_heads=32,
    depth=36,
    ff_size=4096,
)

_bert_oppo_10b = dict(
    seq_length=512,
    hidden_size=4096,
    num_heads=32,
    depth=50,
    ff_size=16384,
)

_bert_24b = dict(
    seq_length=512,
    hidden_size=6144,
    num_heads=48,
    depth=52,
    ff_size=24576,
)

_bert_configurations = dict(
    bert=_bert_base,
    bert_base=_bert_base,
    bert_large=_bert_large,
    bert_oppo=_bert_oppo,
    bert_oppo_10b=_bert_oppo_10b,
    bert_24b=_bert_24b,
)

_default_hyperparameters = dict(
    batch_size=8,
    learning_rate=5e-5,
    weight_decay=1e-2,
    mlm_prob=0.15,
    num_epochs=20,
)


def load_bert_config():
    config = get_config()

    model_type = config['model']['type']
    if model_type in _bert_configurations:
        for k, v in _bert_configurations[model_type].items():
            if k not in config['model']:
                config['model'][k] = v

    if 'hyperparameter' in config:
        for k, v in _default_hyperparameters.items():
            if k not in config['hyperparameter']:
                config['hyperparameter'][k] = v
    else:
        config['hyperparameter'] = _default_hyperparameters

    if 'zero' in config:
        config['zero']['model_config']['shard_strategy'] = TensorShardStrategy()

    if 'fp16' in config:
        config['fp16']['mode'] = AMP_TYPE.NAIVE

    if 'global_batch_size' in config['hyperparameter'] and 'batch_size' in config['hyperparameter']:
        global_bs = config['hyperparameter']['global_batch_size']
        micro_bs = config['hyperparameter']['batch_size']
        accum_size = global_bs // (micro_bs * gpc.data_parallel_size)
        config['gradient_accumulation'] = accum_size

    gpc.load_config(config=config)


def build_data():
    args = get_args()
    config = get_config()

    dataset = load_from_disk(args.data_path)
    tokenizer = BertTokenizer(vocab_file=args.vocab_file)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                    mlm=True,
                                                    mlm_probability=config['hyperparameter']['mlm_prob'])
    global_batch_size = config['hyperparameter'].get('global_batch_size', config['hyperparameter']['batch_size'])
    total_samples = max(global_batch_size * config['hyperparameter'].get('steps_per_epoch', 1), len(dataset['train']))
    multiple = math.ceil(total_samples / len(dataset['train']))

    def repeat(examples):
        result = dict()
        for k, v in examples.items():
            result[k] = np.repeat(v, multiple, axis=0)
        return result

    train_data = dataset['train'].map(repeat, batched=True, load_from_cache_file=False,
                                      keep_in_memory=True).with_format("torch")
    print_log(f'Train dataset loaded:\n{train_data}')

    train_data = get_dataloader(train_data,
                                shuffle=True,
                                drop_last=True,
                                batch_size=config['hyperparameter']['batch_size'],
                                collate_fn=data_collator,
                                num_workers=2,
                                pin_memory=True)

    test_data = None
    if args.do_eval:
        test_data = dataset['test'].with_format("torch")
        print_log(f'Test dataset loaded:\n{test_data}')
        test_data = get_dataloader(dataset['test'],
                                   shuffle=False,
                                   drop_last=True,
                                   batch_size=config['hyperparameter']['batch_size'],
                                   collate_fn=data_collator,
                                   num_workers=2,
                                   pin_memory=True)

    vocab_multiple = gpc.get_world_size(ParallelMode.TENSOR) * 128
    config['model']['vocab_size'] = math.ceil(len(tokenizer) / vocab_multiple) * vocab_multiple

    return train_data, test_data


def build_model():
    config = get_config()
    model_cfg = config['model']

    bert_cfg = BertConfig(
        vocab_size=model_cfg['vocab_size'],
        hidden_size=model_cfg['hidden_size'],
        num_hidden_layers=model_cfg['depth'],
        num_attention_heads=model_cfg['num_heads'],
        intermediate_size=model_cfg['ff_size'],
        max_position_embeddings=model_cfg['seq_length'],
        flash_attention=model_cfg.get('flash_attention', False),
        checkpoint=config.get('gradient_checkpoint', False),
        # use_cache=not config['model'].get('checkpoint', False),
    )
    if 'zero' in config:
        with ZeroInitContext(target_device=get_current_device(), shard_strategy=TensorShardStrategy(), shard_param=True):
            model = ModelFromHF(bert_cfg, BertForMaskedLM)
            if 'numel' not in config['model']:
                config['model']['numel'] = get_model_size(model)

    else:
        model = ModelFromHF(bert_cfg, BertForMaskedLM)
        if 'numel' not in config['model']:
            config['model']['numel'] = get_model_size(model)

    numel = config['model']['numel']
    if numel < 1e9:
        msg = f'{numel / 1e6:.3f} M'
    else:
        msg = f'{numel / 1e9:.3f} B'
    print_log(f'Model is built (parameter size = {msg}).')

    return model


def build_loss():
    return BertMaskedLMLoss()


def build_optimizer(model):
    config = get_config()
    params = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm']
    # configure the weight decay for bert models
    grouped_params = [{
        'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01
    }, {
        'params': [p for n, p in params if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0
    }]
    optim_cls = HybridAdam if 'zero' in config else FusedAdam
    optimizer = optim_cls(grouped_params, lr=config['hyperparameter']['learning_rate'])
    return optimizer


def build_scheduler(dataloader, optimizer):
    config = get_config()
    num_steps = config['hyperparameter'].get('steps_per_epoch', len(dataloader))
    max_steps = config['hyperparameter']['num_epochs'] * num_steps
    warmup_steps = config['hyperparameter'].get('warmup_steps', 0)
    min_lr = config['hyperparameter'].get('min_lr', 0.0)

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(min_lr, float(max_steps - current_step) / float(max(1, max_steps - warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, -1)


def setup_jit_fusion():
    """Set PyTorch JIT layer fusion options."""
    # flags required to enable jit fusion kernels
    torch._C._jit_set_profiling_executor(True)
    torch._C._jit_set_profiling_mode(True)
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_nvfuser_enabled(True)
    torch._C._debug_set_autodiff_subgraph_inlining(False)
    """ Compilie JIT functions before the main training steps """
    config = get_config()
    if 'fp16' in config:
        dtype = torch.float16
    else:
        dtype = torch.float32

    model_cfg = config['model']
    vocab_size = 32768
    hidden_size = model_cfg['hidden_size']
    intermediate_size = model_cfg['ff_size']
    seq_length = model_cfg['seq_length']
    micro_batch_size = config['hyperparameter']['batch_size']

    embed = col_nn.Embedding(vocab_size, hidden_size).to(dtype).to(get_current_device())
    linear_1 = col_nn.Linear(hidden_size, intermediate_size, skip_bias_add=True).to(dtype).to(get_current_device())
    linear_2 = col_nn.Linear(intermediate_size, hidden_size, skip_bias_add=True).to(dtype).to(get_current_device())

    x = torch.randint(vocab_size, (micro_batch_size, seq_length), dtype=torch.long, device=get_current_device())
    x = embed(x)
    y, y_bias = linear_1(x)
    z, z_bias = linear_2(y)
    # Warmup JIT fusions with the input grad_enable state of both forward
    # prop and recomputation
    for bias_grad, input_grad in zip([True, True], [False, True]):
        for _ in range(10):
            bias = torch.rand_like(y_bias, dtype=dtype, device=get_current_device())
            input_ = torch.rand_like(y, dtype=dtype, device=get_current_device())
            bias.requires_grad, input_.requires_grad = bias_grad, input_grad
            bias_gelu(bias, input_)

    # Warmup fused bias+dropout+add
    dropout_rate = 0.1
    # Warmup JIT fusions with the input grad_enable state of both forward
    # prop and recomputation
    for input_grad, bias_grad, residual_grad in zip([False, True], [True, True], [True, True]):
        for _ in range(10):
            input_ = torch.rand_like(z, dtype=dtype, device=get_current_device())
            residual = torch.rand_like(x, dtype=dtype, device=get_current_device())
            bias = torch.rand_like(z_bias, dtype=dtype, device=get_current_device())
            input_.requires_grad = input_grad
            bias.requires_grad = bias_grad
            residual.requires_grad = residual_grad
            bias_dropout_add(input_, bias, residual, dropout_rate, True)

    torch.cuda.empty_cache()


def init_w_col():
    disable_existing_loggers()

    args = get_args()
    config = get_config()

    colossalai.launch_from_torch(config=config)

    logger = get_dist_logger()
    if args.log_path:
        if not os.path.exists(args.log_path):
            os.makedirs(args.log_path, exist_ok=True)
        logger.log_to_file(args.log_path)

    load_bert_config()

    setup_jit_fusion()

    print_log('Building data')
    train_data, test_data = build_data()

    reset_peak_memory_stats()

    print_log('Building model')
    model = build_model()

    criterion = build_loss()

    optimizer = build_optimizer(model)

    lr_scheduler = build_scheduler(train_data, optimizer)

    engine, train_data, test_data, lr_scheduler = colossalai.initialize(model, optimizer, criterion, train_data, test_data,
                                                                        lr_scheduler)
    model = engine.model
    criterion = engine.criterion
    optimizer = engine.optimizer

    return model, train_data, test_data, criterion, optimizer, lr_scheduler
