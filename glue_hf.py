import argparse
import logging
import os
from functools import partial

import evaluate
import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset, load_from_disk
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

PRETRAINED_BERT_NAME = "bert-large-cased"


def tokenize(examples, tokenizer):
    args = (examples["question1"], examples["question2"])
    result = tokenizer(*args, padding="max_length", max_length=128, truncation=True)
    result["label"] = examples["label"]
    return result


def compute_result(p, metric):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    result = metric.compute(predictions=preds, references=p.label_ids)
    if len(result) > 1:
        result["combined_score"] = np.mean(list(result.values())).item()
    return result


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

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO if rank == 0 else logging.CRITICAL)

    config = BertConfig()
    if args.model_config is not None:
        config = config.from_json_file(args.model_config)
    else:
        config = config.from_pretrained(PRETRAINED_BERT_NAME)

    model = BertForSequenceClassification(config)

    if args.model_weight is not None:
        state = torch.load(args.model_weight, map_location="cpu")
        model.load_state_dict(state)
    else:
        model = model.from_pretrained(PRETRAINED_BERT_NAME)

    if args.vocab_file is None:
        tokenizer = BertTokenizer.from_pretrained(PRETRAINED_BERT_NAME)
    else:
        tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=False)

    logger.info("***** Processing dataset *****")
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

    metric = evaluate.load("glue", "qqp")

    if args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    trainer_args = TrainingArguments(output_dir=".",
                                     do_train=False,
                                     do_eval=True,
                                     fp16=args.fp16,
                                     fp16_full_eval=args.fp16,
                                     per_device_eval_batch_size=args.batch_size)

    trainer = Trainer(
        model=model,
        args=trainer_args,
        eval_dataset=dataset,
        compute_metrics=partial(compute_result, metric=metric),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    torch.cuda.reset_peak_memory_stats()
    metrics = trainer.evaluate(eval_dataset=dataset)
    memory = torch.cuda.max_memory_allocated()
    metrics["eval_mem_usage"] = memory
    trainer.log_metrics("eval", metrics)


if __name__ == "__main__":
    main()
