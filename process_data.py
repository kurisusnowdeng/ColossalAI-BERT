import argparse

import nltk
from datasets import load_dataset
from transformers import BertTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--vocab-file", type=str)
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--num-workers", type=int, default=16)
    return parser.parse_args()


def build_dataset():
    args = parse_args()

    print(f"Loading from file {args.data_path} ...")

    dataset = load_dataset("json", data_files=args.data_path)
    dataset = dataset['train'].train_test_split(train_size=0.9, shuffle=False)

    tokenizer = BertTokenizer(vocab_file=args.vocab_file)
    splitter = nltk.load("tokenizers/punkt/english.pickle")

    def tokenize(examples):
        seq_len = 512
        tokenizer.pad_token = tokenizer.unk_token
        sentences = []
        for document in examples['text']:
            for sentence in splitter.tokenize(document):
                sentences.append(sentence)
        result = tokenizer(sentences, padding='max_length', truncation=True, max_length=seq_len, return_tensors='np')

        return result

    remove_keys = list(dataset['train'][0].keys())
    tokenized_dataset = dataset.map(tokenize,
                                    batched=True,
                                    num_proc=args.num_workers,
                                    load_from_cache_file=False,
                                    keep_in_memory=True,
                                    remove_columns=remove_keys)

    print(f"Loaded dataset:\n{tokenized_dataset}")

    print(f"Saving dataset to {args.output_path} ...")
    tokenized_dataset.save_to_disk(args.output_path)
    print("Dataset saved.")


if __name__ == '__main__':
    build_dataset()
