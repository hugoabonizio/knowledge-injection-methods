import os
import argparse
from transformers import AutoTokenizer
from datasets import load_dataset, disable_progress_bars
disable_progress_bars()

parser = argparse.ArgumentParser(
    description="Count the number of tokens on a given dataset."
)
parser.add_argument("--dataset_name", type=str, default="hugo/knowledge-injection-1")
parser.add_argument("--dataset_config", type=str, default="default")
parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-2-7b-chat-hf")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
dataset = load_dataset(args.dataset_name, args.dataset_config)["train"]
dataset = dataset.map(
    lambda x: {"length": len(tokenizer.tokenize(x["text"]))},
    remove_columns=dataset.column_names,
    num_proc=os.cpu_count(),
)
total_tokens = sum(dataset["length"])
print(f"Total number of tokens: {total_tokens:_}")
print(f"Average: {round(total_tokens / len(dataset)):_} | Max: {max(dataset['length']):_}")
