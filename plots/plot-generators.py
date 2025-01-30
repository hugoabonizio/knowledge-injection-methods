import os
import json
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt
from statistics import mean, stdev
import matplotlib.patches as mpatches

def read_result(news_result_path):
    acc_new_list = []

    if "*" in news_result_path:
        files = list(glob(news_result_path))
    else:
        files = [news_result_path]

    for file in files:
        df = pd.read_json(file, lines=True)
        df["year"] = df["year"].astype(int)
        # old = year <= 2022; new = year > 2022
        df_new = df.query("year > 2022")
        correct_new = sum(df_new["result"].str.match("yes"))
        acc_new_list.append(correct_new / len(df_new) * 100)

    mean_new = round(mean(acc_new_list), 1)
    return mean_new

cmap = plt.get_cmap('Dark2')
colors = [cmap(0), cmap(0, alpha=0.7), cmap(0, alpha=0.3), cmap(2)]

rtw_gpt4 = read_result(
    'outputs/results/llama2_7b_chat_rephrasing_prompts4_repeat40_temperature10_gpt4o_*_closedbook_evaluations.json'
)
rtw_llama2 = read_result(
    'outputs/results/llama2_7b_chat_rephrasing_prompts4_repeat40_temperature10_llama2_*_closedbook_evaluations.json'
)
rtw_gpt4_noqa = read_result(
    'outputs/results/llama2_7b_chat_rephrasing_prompts3_repeat40_temperature10_gpt4o_*_closedbook_evaluations.json'
)
rtw_llama2_noqa = read_result(
    'outputs/results/llama2_7b_chat_rephrasing_prompts3_repeat40_temperature10_llama2_*_closedbook_evaluations.json'
)
rtw_gpt4_qaonly = read_result(
    'outputs/results/llama2_7b_chat_rephrasing_prompts1_repeat40_temperature10_gpt4o_*_closedbook_evaluations.json'
)
rtw_llama2_qaonly = read_result(
    'outputs/results/llama2_7b_chat_rephrasing_prompts1_repeat40_temperature10_llama2_*_closedbook_evaluations.json'
)

para_gpt4 = read_result(
    'outputs/results/llama2_7b_chat_paraphrases_repeat40-temperature_10-gpt4o_closedbook_evaluations.json'
)
para_llama2 = read_result(
    'outputs/results/llama2_7b_chat_paraphrases_repeat40-temperature_10-llama2_closedbook_evaluations.json'
)

groups = ['RTW', 'RTW (QA-only)', 'RTW (no QA)', 'Para']
models = ['GPT-4o', 'Llama-2']
data = {
    'RTW': [
        rtw_gpt4,
        rtw_llama2,
    ],
    'RTW (QA-only)': [
        rtw_gpt4_qaonly,
        rtw_llama2_qaonly,
    ],
    'RTW (no QA)': [
        rtw_gpt4_noqa,
        rtw_llama2_noqa,
    ],
    'Para': [
        para_gpt4,
        para_llama2,
    ]
}

x = np.arange(len(groups))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 4), dpi=300)

for i, group in enumerate(groups):
    bar = ax.bar(
        x[i] - width/2,
        data[group][0],
        width,
        color=colors[i],
        edgecolor='black',
        linewidth=1,
        label=data[group][0],
    )
    ax.bar_label(bar, padding=3, fmt='{:.1f}')
    bar = ax.bar(
        x[i] + width/2,
        data[group][1],
        width,
        color=colors[i],
        hatch='//',
        edgecolor='black',
        linewidth=1,
        label=data[group][0],
    )
    ax.bar_label(bar, padding=3, fmt='{:.1f}')

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=12)
ax.set_ylim(30, 80)

group_patches = [
    mpatches.Patch(facecolor=colors[0], edgecolor='black', label='RTW'),
    mpatches.Patch(facecolor=colors[1], edgecolor='black', label='RTW (no QA)'),
    mpatches.Patch(facecolor=colors[2], edgecolor='black', label='RTW (QA-only)'),
    mpatches.Patch(facecolor=colors[3], edgecolor='black', label='Para'),
]

model_patches = [
    mpatches.Patch(facecolor='white', edgecolor='black', hatch='', label='GPT-4o'),
    mpatches.Patch(facecolor='white', edgecolor='black', hatch='//', label='Llama-2'),
]

all_handles = group_patches + model_patches
ax.legend(handles=all_handles, fontsize=10)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.grid(True, linestyle='-', alpha=0.1, axis='y')
plt.tight_layout()

plt.savefig('./plots/plot-generators.png', bbox_inches='tight')
