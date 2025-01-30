import os
import json
import pandas as pd
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from statistics import mean, stdev
from transformers import AutoTokenizer
from datasets import load_dataset, disable_progress_bars
disable_progress_bars()
from lm_eval.utils import (
    get_latest_filename,
    get_results_filenames,
)


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

def read_lmeval(results_path):
    results_dir = Path(results_path)
    files = [f.as_posix() for f in results_dir.iterdir() if f.is_file()]
    results_filenames = get_results_filenames(files)
    latest_results = get_latest_filename(results_filenames)
    with open(latest_results) as f:
        results = json.load(f)
        accs = []
        for result, metrics in results["results"].items():
            accs.append(metrics["acc,none"])
        return mean(accs) * 100


def count_tokens(dataset_name, dataset_config):
    dataset = load_dataset(dataset_name, dataset_config)["train"]
    dataset = dataset.map(
        lambda x: {"length": len(tokenizer.tokenize(x["text"]))},
        remove_columns=dataset.column_names,
        num_proc=os.cpu_count(),
    )
    return sum(dataset["length"])

# cmap = plt.get_cmap('Set1')
cmap = plt.get_cmap('Dark2')
colors = [cmap(i) for i in range(10)]
rtw_qaonly = cmap(0, alpha=0.7)
rtw_noqa = cmap(0, alpha=0.3)

variations = [1, 5, 10, 20, 40]
rag_cmap = plt.get_cmap('tab20c')

results = [
    {
        "name": "IPT",
        "ticks": variations,
        "color": colors[1],
        "linewidth": 2,
        "acc": [
            read_lmeval(f"outputs/control/outputs__models__llama2_7b_chat_instruction_repeat{n_repeats}-temperature_{'00' if n_repeats == 1 else '10'}_1/")
            for n_repeats in variations
        ],
        "tokens": [
            count_tokens('hugo/knowledge-injection-1', f"instruction-1_shot-repeat_{n_repeats}-temperature_{'00' if n_repeats == 1 else '10'}")
            for n_repeats in variations
        ],
    },
    {
        "name": "Para",
        "ticks": variations,
        "color": colors[2],
        "linewidth": 2,
        "acc": [
            read_lmeval(f'outputs/control/outputs__models__llama2_7b_chat_paraphrases_repeat{n_repeats}-temperature_10-gpt4o/')
            for n_repeats in variations
        ],
        "tokens": [
            count_tokens('hugo/knowledge-injection-1', f"paraphrases-repeat_{n_repeats}-temperature_10-gpt4o")
            for n_repeats in variations
        ],
    },
    {
        "name": "RTW",
        "ticks": variations,
        "color": colors[0],
        "linewidth": 2,
        "acc": [
            read_lmeval(f'outputs/control/outputs__models__llama2_7b_chat_rephrasing_prompts4_repeat{n_repeats}_temperature10_gpt4o_1/')
            for n_repeats in variations
        ],
        "tokens": [
            count_tokens('hugo/knowledge-injection-1', f"rephrasing-prompts_4-repeat_{n_repeats}-temperature_10-gpt4o")
            for n_repeats in variations
        ],
    },
    {
        "name": "RTW (QA-only)",
        "ticks": variations,
        "color": rtw_qaonly,
        "linewidth": 2,
        "linestyle": 'dashed',
        "acc": [
            read_lmeval(f'outputs/control/outputs__models__llama2_7b_chat_rephrasing_prompts1_repeat{n_repeats}_temperature10_gpt4o_1/')
            for n_repeats in variations
        ],
        "tokens": [
            count_tokens('hugo/knowledge-injection-1', f"rephrasing-prompts_1-repeat_{n_repeats}-temperature_10-gpt4o")
            for n_repeats in variations
        ],
    },
    {
        "name": "RTW (no QA)",
        "ticks": variations,
        "color": rtw_noqa,
        "linewidth": 2,
        "linestyle": 'dashed',
        "acc": [
            read_lmeval(f'outputs/control/outputs__models__llama2_7b_chat_rephrasing_prompts3_repeat{n_repeats}_temperature10_gpt4o_1/')
            for n_repeats in variations
        ],
        "tokens": [
            count_tokens('hugo/knowledge-injection-1', f"rephrasing-prompts_3-repeat_{n_repeats}-temperature_10-gpt4o")
            for n_repeats in variations
        ],
    },
    {
        "name": 'CPT',
        "color": 'black',
        "marker": "*",
        "s": 80,
        "acc": read_lmeval('outputs/control/outputs__models__llama2_7b_chat_cpt/'),
        "tokens": count_tokens('hugo/knowledge-injection-1', 'default'),
    },
    {
        "name": 'Original',
        "color": 'black',
        "linewidth": 1,
        "linestyle": 'dashed',
        "acc": read_lmeval('outputs/control/meta-llama__Llama-2-7b-chat-hf/'),
    },
    {
        "name": 'RAG (Doc)',
        "color": rag_cmap(17),
        "linewidth": 2,
        "acc": read_lmeval('outputs/control_rag/top1/meta-llama__Llama-2-7b-chat-hf/'),
    },
    {
        "name": 'RAG (Chunk)',
        "color": rag_cmap(18),
        "linewidth": 2,
        "acc": read_lmeval('outputs/control_rag/top5/meta-llama__Llama-2-7b-chat-hf/'),
    },
]

plt.figure(figsize=(7, 4), dpi=300)

for result in results:
    if isinstance(result['acc'], list):
        plt.plot(
            result['tokens'],
            result['acc'],
            marker='o',
            color=result['color'],
            linewidth=result.get('linewidth', 2),
            label=result['name'],
            linestyle=result.get('linestyle'),
        )
    elif result.get('marker', None):
        plt.scatter(
            result['tokens'],
            result['acc'],
            marker=result.get('marker'),
            color=result['color'],
            label=result['name'],
            s=result.get('s', 2),
        )
    else:
        plt.axhline(
            result['acc'],
            color=result['color'],
            linewidth=result.get('linewidth', 1),
            label=result['name'],
            linestyle=result.get('linestyle'),
        )

# plt.xlim([0, 41])
plt.ylim([60.5, 63])
# plt.xticks(xticks)
plt.xlabel('Training tokens', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.grid(True, linestyle='-', alpha=0.1)
plt.legend(loc='lower right', fontsize=8, title='Methods', bbox_to_anchor=(1.3, 0.5))

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xscale('log')
ax.set_xticks([250_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000])
ax.xaxis.set_major_formatter(ticker.EngFormatter())

plt.tight_layout()
plt.savefig('./plots/plot-control-datasets.png', bbox_inches='tight')