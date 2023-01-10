import argparse
import os
import time
from functools import partial

import colossalai
import colossalai.nn as col_nn
import torch
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.utils import get_dataloader
from datasets import concatenate_datasets, load_dataset, load_from_disk
from model import BertForSequenceClassification
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer, DataCollatorWithPadding

PRETRAINED_BERT_NAME = "bert-large-cased"


def tokenize(examples, tokenizer):
    args = (examples["question1"], examples["question2"])
    result = tokenizer(*args, padding="max_length", max_length=128, truncation=True)
    result["label"] = examples["label"]
    return result


def process_weight(config, state_dict):
    state_dict.pop("bert.embeddings.position_ids")
    # concat qkv params
    for i in range(config.num_hidden_layers):
        prefix = f"bert.encoder.layer.{i}.attention.self"
        names = ["query", "key", "value"]
        for p in ["weight", "bias"]:
            params = [state_dict.pop(".".join((prefix, n, p))) for n in names]
            key = ".".join((prefix, "_".join(names), p))
            state_dict[key] = torch.cat(params, dim=0)
    # rename pooler params
    state_dict["pooler.dense.weight"] = state_dict.pop("bert.pooler.dense.weight")
    state_dict["pooler.dense.bias"] = state_dict.pop("bert.pooler.dense.bias")

    return state_dict


def compile_model(model, dataloader, dtype):
    example_inputs = next(iter(dataloader))
    example_inputs.pop("labels")
    example_inputs = tuple([torch.ones_like(example_inputs[key]).cuda() for key in example_inputs])
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        jit_model = torch.jit.trace(model.eval(), example_inputs, strict=False)
        jit_model = torch.jit.optimize_for_inference(jit_model)

    return jit_model


def get_time():
    torch.cuda.synchronize()
    torch.distributed.barrier()
    return time.time()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config', type=str)
    parser.add_argument('--model-weight', type=str)
    parser.add_argument('--vocab-file', type=str)
    parser.add_argument('--data-path', type=str)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--fp16', action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    disable_existing_loggers()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    colossalai.launch_from_torch({})

    logger = get_dist_logger()
    log = partial(logger.info, ranks=[0])

    config = BertConfig()
    if args.model_config is not None:
        config = config.from_json_file(args.model_config)
    else:
        config = config.from_pretrained(PRETRAINED_BERT_NAME)
    config.flash_attention = False

    model = BertForSequenceClassification(config)

    if args.model_weight is not None:
        state = torch.load(args.model_weight, map_location="cpu")
        state = process_weight(config, state)
        model.load_state_dict(state)
    model = model.cuda()
    if args.fp16:
        model = model.half()

    if args.vocab_file is None:
        tokenizer = BertTokenizer.from_pretrained(PRETRAINED_BERT_NAME)
    else:
        tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=False)

    log("***** Processing dataset *****")
    if args.data_path is not None:
        dataset = load_from_disk(args.data_path)
        dataset = dataset["validation"]
    else:
        dataset = load_dataset("glue", "qqp", split="validation")
    dataset = concatenate_datasets([dataset] * 5)
    dataset = dataset.map(partial(tokenize, tokenizer=tokenizer),
                          batched=True,
                          num_proc=args.num_workers,
                          load_from_cache_file=False,
                          keep_in_memory=True,
                          remove_columns=dataset.column_names)

    dataloader = get_dataloader(dataset,
                                batch_size=args.batch_size,
                                collate_fn=DataCollatorWithPadding(tokenizer),
                                pin_memory=True,
                                num_workers=args.num_workers)

    dtype = torch.float16 if args.fp16 else torch.float32

    model = compile_model(model, dataloader, dtype)

    criterion = col_nn.CrossEntropyLoss()

    progress = range(len(dataloader))
    if rank == 0:
        progress = tqdm(progress)

    log("***** Running Evaluation *****")
    log(f"  Num examples = {len(dataset)}")
    log(f"  Batch size per device = {args.batch_size}")
    log(f"  Global batch size = {args.batch_size * world_size}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = get_time()
    loss = 0.
    data_iterator = iter(dataloader)
    for i in progress:
        if i == 2:
            start = get_time()

        batch = next(data_iterator)
        for k, v in batch.items():
            batch[k] = v.cuda()
        labels = batch.pop("labels")

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
            logits = model(**batch)["logits"]

        loss += criterion(logits, labels).item()

    end = get_time()

    memory = torch.cuda.max_memory_allocated() / 1024**2

    log("***** Evaluation Metrics *****")
    log(f" eval_loss = {loss / len(dataloader):.4f}")
    log(f" eval_mem_usage = {memory:.0f} MB")
    log(f" eval_runtime = {(end - start):.2f} s")
    step_time = (end - start) / (len(dataloader) - 2)
    log(f" eval_steps_per_second = {1 / step_time:.3f}")
    throughput = args.batch_size * world_size / step_time
    log(f" eval_samples_per_second = {throughput:.3f}")


if __name__ == "__main__":
    main()
