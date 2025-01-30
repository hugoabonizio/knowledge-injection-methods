import os
import json
import torch
import argparse
from tqdm import trange
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)
import bm25s
import Stemmer
from llama_index.core.node_parser import SentenceSplitter


parser = argparse.ArgumentParser(description="Generate answers with a model.")
parser.add_argument(
    "--model_name", type=str, required=True, help="Name or path of the model."
)
parser.add_argument(
    "--output", type=str, required=True, help="Path to the output JSON file."
)
parser.add_argument("--dataset_name", type=str, default="hugo/news-corpus-1")
parser.add_argument(
    "--context", type=str, default=None, choices=["oracle", "top1", "top5"]
)
parser.add_argument("--max_new_tokens", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

set_seed(1)

model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    torch_dtype=torch.bfloat16,
)
model.to("cuda")
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

if tokenizer.chat_template is None:
    raise ValueError(
        "Only chat models are supported. The tokenizer has no 'chat_template' attribute."
    )

dataset = load_dataset(args.dataset_name)["test"]

if args.context:
    prompt_template = """Answer the question based on the following context:
{context}

Q: {question}"""
else:
    prompt_template = "{question}"


def build_prompt(question, context=None):
    if context is None:
        prompt = prompt_template.format(question=question)
    else:
        prompt = prompt_template.format(
            context=context,
            question=question,
        )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
    ).strip()


if args.context:
    if args.context.startswith("top"):
        stemmer = Stemmer.Stemmer("english")
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
        if args.context == "top1":
            corpus = list(set([
                example["context"]
                for example in dataset
                if example["year"] in ["2023", "2024"]
            ]))
        elif args.context == "top5":
            corpus = sum(
                [
                    splitter.split_text(example["context"])
                    for example in dataset
                    if example["year"] in ["2023", "2024"]
                ],
                [],
            )
        print(f'Corpus size: {len(corpus):=}')
        corpus_tokens = bm25s.tokenize(corpus, stopwords='en', stemmer=stemmer)
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)


def get_context(example):
    if args.context == "oracle":
        return example["context"]
    elif args.context.startswith("top"):
        query_tokens = bm25s.tokenize(example["question"], stemmer=stemmer)
        if args.context == "top1":
            results, scores = retriever.retrieve(query_tokens, k=1)
        elif args.context == "top5":
            results, scores = retriever.retrieve(query_tokens, k=5)
        return '\n\n'.join([corpus[doc_idx[0]] for doc_idx in results])

examples = [
    {
        "prompt": (
            build_prompt(example["question"], get_context(example))
            if args.context
            else build_prompt(example["question"])
        ),
        "question": example["question"],
        "answer": example["answer"],
        "year": example["year"],
    }
    for example in dataset
]

out_dir = os.path.dirname(args.output)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

with open(args.output, "w") as out:
    for batch_idx in trange(0, len(examples), args.batch_size, desc="Generating"):
        batch = examples[batch_idx : batch_idx + args.batch_size]
        prompts = [example["prompt"] for example in batch]
        tokens = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")

        output = model.generate(
            **tokens,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            stop_strings=["\n"],
            do_sample=False,
        )
        answers = tokenizer.batch_decode(
            output[:, tokens["input_ids"].shape[1] :], skip_special_tokens=True
        )
        for example, answer in zip(batch, answers):
            out.write(
                json.dumps(
                    {
                        "question": example["question"],
                        "answer_true": example["answer"],
                        "answer_model": answer.strip(),
                        "year": example["year"],
                    }
                )
                + "\n"
            )
