import json
import torch
from typing import Optional
from datasets import load_dataset
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)
from trl import SFTConfig, SFTTrainer
from dataclasses import dataclass

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


@dataclass
class CustomArguments:
    model_name_or_path: str
    dataset_name: str
    dataset_config_name: str = "default"


def train():
    parser = HfArgumentParser((SFTConfig, CustomArguments))
    training_args, custom_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        custom_args.model_name_or_path,
        use_fast=False,
        model_max_length=training_args.max_seq_length,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = load_dataset(custom_args.dataset_name, custom_args.dataset_config_name)
    train_dataset = dataset["train"]

    model = AutoModelForCausalLM.from_pretrained(
        custom_args.model_name_or_path,
        use_cache=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    train()
