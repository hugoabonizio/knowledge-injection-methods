# Adapted from: https://github.com/microsoft/LMOps/blob/main/instruction_pretrain/README.md

import json
import argparse
from itertools import chain
from datasets import load_dataset
from vllm import LLM, SamplingParams
from utils.read_compre import get_dataset, cook_pt_entries, run

parser = argparse.ArgumentParser(
    description="Generate augmentation using the Instruction Pretrain techinque."
)
parser.add_argument("--dataset_name", type=str, default="hugo/knowledge-injection-1")
parser.add_argument("--dataset_config", type=str, default="default")
parser.add_argument("--repeat", type=int, default=1)
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

dataset = load_dataset(args.dataset_name, args.dataset_config)["train"]
raw_texts = [example["text"] for example in dataset]

N = len(raw_texts)  # Number of raw texts
M = 1  # M-shot example
max_model_len = 4096  # max squence len of the LM you intend to pre-train
max_new_tokens = (
    400  # max number of tokens for the augmented instruction-response pairs
)

temperature = 0 if args.repeat == 1 else 1.0
sampling_params = SamplingParams(temperature=temperature, max_tokens=max_new_tokens)

llm = LLM(
    model="instruction-pretrain/instruction-synthesizer",
    max_model_len=max_model_len,
)

all_generated_examples = []

for repeat in range(args.repeat):
    # 1. multi-round inference to get the prediction
    prev_examples = []
    BSZ = (N + M - 1) // M
    for round in range(M):
        cur_raw_texts = raw_texts[round * BSZ : (round + 1) * BSZ]
        # load data
        split = get_dataset(
            prev_examples=prev_examples,
            cur_raw_texts=cur_raw_texts,
            max_model_len=max_model_len,
            max_new_tokens=max_new_tokens,
        )
        prev_examples = run(split, llm, sampling_params)


    # 2. templify the data for subsequent pre-training
    instruction_augmented_texts = []
    for idx, entry in enumerate(prev_examples):
        texts = cook_pt_entries(read_collection=entry, random_seed=idx + 12345 + repeat)
        # change random seed for each entry for diveristy
        instruction_augmented_texts.extend(texts)

    all_generated_examples += instruction_augmented_texts

with open(args.output, "w") as out:
    for text in chain(all_generated_examples):
        out.write(json.dumps({"text": text}) + "\n")
