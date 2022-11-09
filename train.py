import math
import time

import numpy as np
import torch
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.utils.cuda import get_current_device
from torch.distributed import all_reduce
from tqdm import tqdm

from utils import get_args, get_config, get_tflops, print_log

model_mem = None


def aggregate_ddp_results(*vals):
    tensor = torch.as_tensor(vals, device=get_current_device())
    all_reduce(tensor, group=gpc.get_group(ParallelMode.DATA))
    return tuple(tensor.tolist())


def _train(epoch, train_dataloader, model, criterion, optimizer, lr_scheduler):
    config = get_config()
    rank = gpc.get_global_rank()
    world_size = gpc.get_world_size(ParallelMode.DATA)
    fp16 = 'fp16' in config

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model.train()

    num_steps = len(train_dataloader)
    accum_size = config.get('gradient_accumulation', 1)
    global_batch_size = config['hyperparameter']['batch_size'] * accum_size * world_size
    seq_length = config['model']['seq_length']
    num_steps = num_steps // accum_size
    if 'steps_per_epoch' in config['hyperparameter']:
        num_steps = config['hyperparameter']['steps_per_epoch']
    progress = range(num_steps)

    if rank == 0:
        progress = tqdm(progress, desc=f"[Epoch {epoch} / Train]")

    train_loss = list()
    num_tokens = list()

    data_iter = iter(train_dataloader)
    used_time = list()

    for _ in progress:
        data_time = 0.
        fwd_time = 0.
        bwd_time = 0.
        opt_time = 0.
        batch_tokens = 0
        for _ in range(accum_size):

            data_start = time.time()
            batch = next(data_iter)
            for k, v in batch.items():
                batch[k] = v.cuda()
            labels = batch.pop('labels')
            batch_tokens += labels.numel()
            data_end = time.time()

            fwd_start = data_end
            outputs = model(**batch)
            loss = criterion(outputs, labels)
            train_loss.append(loss.item())
            fwd_end = time.time()

            bwd_start = fwd_end
            optimizer.backward(loss)
            torch.cuda.synchronize()
            bwd_end = time.time()

            opt_start = bwd_end
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            torch.cuda.synchronize()
            opt_end = time.time()

            data_time += data_end - data_start
            fwd_time += fwd_end - fwd_start
            bwd_time += bwd_end - bwd_start
            opt_time += opt_end - opt_start

        batch_time = data_time + fwd_time + bwd_time + opt_time
        used_time.append(batch_time)
        num_tokens.append(batch_tokens)
        if rank == 0:
            states = dict(loss=loss.item(),
                          lr=lr_scheduler.get_last_lr()[0],
                          time_dataloader=data_time,
                          time_forward=fwd_time,
                          time_backward=bwd_time,
                          time_optimizer=opt_time,
                          throughput=global_batch_size / batch_time,
                          tflops=get_tflops(batch_time))
            if fp16:
                states['scale'] = optimizer.optim.optim.loss_scale.item()
            progress.set_postfix(**states)

    used_time = np.sum(used_time[1:])
    peak_mem = torch.cuda.max_memory_allocated()
    state_mem = torch.cuda.memory_allocated() - model_mem
    activation_mem = peak_mem - state_mem - model_mem

    train_loss = np.sum(train_loss)
    num_tokens = np.sum(num_tokens[1:])
    train_loss, num_tokens = aggregate_ddp_results(train_loss, num_tokens)

    msg = f'[Epoch {epoch} / Train]: Loss = {train_loss / (world_size * num_steps * accum_size):.3f}'
    msg += f' | Throughput = {num_tokens / used_time:.3f} tokens/sec ({num_tokens / (used_time * seq_length):.3f} samples/sec)'
    msg += f' | TFLOPS = {get_tflops(used_time / (num_steps - 1)):.3f}'

    msg += f"\n[Epoch {epoch} / Train]: Peak memory = {peak_mem / 1024**3:.3f} GB"
    msg += f" | Model memory = {model_mem / 1024**3:.3f} GB."
    msg += f" | Optimizer state memory = {state_mem / 1024**3:.3f} GB."
    msg += f" | Activation memory = {activation_mem / 1024**3:.3f} GB."
    print_log(msg)


def _test(test_dataloader, model, criterion):
    config = get_config()
    rank = gpc.get_global_rank()
    world_size = gpc.get_world_size(ParallelMode.DATA)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model.eval()

    num_steps = len(test_dataloader)
    accum_size = config.get('gradient_accumulation', 1)
    global_batch_size = config['hyperparameter']['batch_size'] * accum_size * world_size
    seq_length = config['model']['seq_length']
    num_steps = num_steps // accum_size
    if 'test_steps' in config['hyperparameter']:
        num_steps = config['hyperparameter']['test_steps']
    progress = range(num_steps)

    if rank == 0:
        progress = tqdm(progress, desc="[Test]")

    test_loss = list()
    num_tokens = list()

    data_time = 0.
    fwd_time = 0.

    data_iter = iter(test_dataloader)
    used_time = list()

    with torch.no_grad():
        for _ in progress:
            data_time = 0.
            fwd_time = 0.
            batch_tokens = 0
            for _ in range(accum_size):
                data_start = time.time()
                batch = next(data_iter)
                for k, v in batch.items():
                    batch[k] = v.cuda()
                labels = batch.pop('labels')
                batch_tokens += labels.numel()
                data_end = time.time()

                fwd_start = data_end
                outputs = model(**batch)
                loss = criterion(outputs, labels)
                test_loss.append(loss.item())
                torch.cuda.synchronize()
                fwd_end = time.time()

                data_time += data_end - data_start
                fwd_time += fwd_end - fwd_start

            batch_time = data_time + fwd_time
            used_time.append(batch_time)
            num_tokens.append(batch_tokens)
            if rank == 0:
                progress.set_postfix(loss=loss.item(),
                                     ppl=math.exp(loss.item()),
                                     time_dataloader=data_time,
                                     time_forward=fwd_time,
                                     throughput=global_batch_size / batch_time,
                                     tflops=get_tflops(batch_time))

    used_time = np.sum(used_time[1:])
    peak_mem = torch.cuda.max_memory_allocated()
    activation_mem = peak_mem - model_mem

    test_loss = np.sum(test_loss)
    num_tokens = np.sum(num_tokens[1:])
    test_loss, num_tokens = aggregate_ddp_results(test_loss, num_tokens)

    msg = f'[Test]: Loss = {test_loss / (world_size * num_steps * accum_size):.3f}'
    msg += f' | Perplexity = {math.exp(test_loss / (world_size * num_steps * accum_size)):.3f}'
    msg += f' | Throughput = {num_tokens / used_time:.3f} tokens/sec ({num_tokens / (used_time * seq_length):.3f} samples/sec)'
    msg += f' | TFLOPS = {get_tflops(used_time / (num_steps - 1)):.3f}'

    msg += f"\n[Test]: Peak memory = {peak_mem / 1024**3:.3f} GB"
    msg += f" | Model memory = {model_mem / 1024**3:.3f} GB."
    msg += f" | Activation memory = {activation_mem / 1024**3:.3f} GB."
    print_log(msg)


def train(model, train_data, test_data, criterion, optimizer, lr_scheduler):
    args = get_args()
    config = get_config()

    global model_mem
    model_mem = torch.cuda.memory_allocated()

    print_log('Benchmark start.')

    for epoch in range(config['hyperparameter']['num_epochs']):
        _train(epoch, train_data, model, criterion, optimizer, lr_scheduler)
    if args.do_eval:
        _test(test_data, model, criterion)

    print_log('Benchmark complete.')
