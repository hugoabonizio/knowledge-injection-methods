import os
import json
import pandas as pd
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt
from statistics import mean, stdev
from transformers import AutoTokenizer
from datasets import load_dataset, disable_progress_bars
disable_progress_bars()


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

def read_result(news_result_path):
    acc_old_list = []
    acc_new_list = []

    if "*" in news_result_path:
        files = list(glob(news_result_path))
    else:
        files = [news_result_path]

    for file in files:
        df = pd.read_json(file, lines=True)
        df["year"] = df["year"].astype(int)
        df_old = df.query("year <= 2022")
        correct_old = sum(df_old["result"].str.match("yes"))
        df_new = df.query("year > 2022")
        correct_new = sum(df_new["result"].str.match("yes"))
        acc_old_list.append(correct_old / len(df_old) * 100)
        acc_new_list.append(correct_new / len(df_new) * 100)

    mean_new = round(mean(acc_new_list), 1)
    std_new = round(stdev(acc_new_list), 1) if len(acc_new_list) > 1 else 0
    return mean_new


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
        "name": 'Open-book (oracle)',
        "color": 'black',
        "linewidth": 1,
        "linestyle": 'dashdot',
        "acc": read_result('outputs/results/llama2_7b_chat_original_openbook_evaluations.json'),
    },
    {
        "name": 'RAG (Doc)',
        "color": rag_cmap(17),
        "linewidth": 2,
        "acc": read_result('outputs/results/llama2_7b_chat_original_ragtop1_evaluations.json'),
    },
    {
        "name": 'RAG (Chunk)',
        "color": rag_cmap(18),
        "linewidth": 2,
        "acc": read_result('outputs/results/llama2_7b_chat_original_ragtop5_evaluations.json'),
    },
    {
        "name": "RTW",
        "ticks": variations,
        "color": colors[0],
        "linewidth": 2,
        "acc": [
            read_result(f'outputs/results/llama2_7b_chat_rephrasing_prompts4_repeat{n_repeats}_temperature10_gpt4o_*_closedbook_evaluations.json')
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
            read_result(f'outputs/results/llama2_7b_chat_rephrasing_prompts1_repeat{n_repeats}_temperature10_gpt4o_*_closedbook_evaluations.json')
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
            read_result(f'outputs/results/llama2_7b_chat_rephrasing_prompts3_repeat{n_repeats}_temperature10_gpt4o_*_closedbook_evaluations.json')
            for n_repeats in variations
        ],
    },
    {
        "name": "Para",
        "ticks": variations,
        "color": colors[2],
        "linewidth": 2,
        "acc": [
            read_result(f'outputs/results/llama2_7b_chat_paraphrases_repeat{n_repeats}-temperature_10-gpt4o_closedbook_evaluations.json')
            for n_repeats in variations
        ],
    },
    {
        "name": 'CPT',
        "color": 'black',
        "linewidth": 2,
        "acc": read_result('outputs/results/llama2_7b_chat_vanilla_closedbook_evaluations.json'),
    },
    {
        "name": "IPT",
        "ticks": variations,
        "color": colors[1],
        "linewidth": 2,
        "acc": [
            read_result(f'outputs/results/llama2_7b_chat_instruction_repeat{n_repeats}-temperature_{"00" if n_repeats == 1 else "10"}_*_closedbook_evaluations.json')
            for n_repeats in variations
        ],
    },
    {
        "name": 'Original',
        "color": 'black',
        "linewidth": 1,
        "linestyle": 'dashed',
        "acc": read_result('outputs/results/llama2_7b_chat_original_closedbook_evaluations.json'),
    },
]

plt.figure(figsize=(7, 4), dpi=300)

xticks = [1, 5, 10, 20, 40]

for result in results:
    if isinstance(result['acc'], list):
        plt.plot(
            result['ticks'],
            result['acc'],
            marker='o',
            color=result['color'],
            linewidth=result.get('linewidth', 2),
            label=result['name'],
            linestyle=result.get('linestyle'),
        )
    else:
        plt.axhline(
            result['acc'],
            color=result['color'],
            linewidth=result.get('linewidth', 1),
            label=result['name'],
            linestyle=result.get('linestyle'),
        )

plt.xlim([0, 41])
plt.ylim([0, 80])
plt.xticks(xticks)
plt.xlabel('Number of variations', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.grid(True, linestyle='-', alpha=0.1)
plt.legend(loc='lower right', fontsize=8, title='Methods', bbox_to_anchor=(1.4, 0.4))

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ax.set_xscale('log')
# ax.set_xticks(xticks)
# ax.set_xlim([0, 41])
# import matplotlib.ticker as ticker
# ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

plt.tight_layout()
plt.savefig('./plots/plot-variations.png', bbox_inches='tight')