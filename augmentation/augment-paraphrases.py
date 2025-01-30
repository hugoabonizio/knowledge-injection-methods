import os
import json
import time
import openai
import argparse
from tqdm import tqdm
from itertools import chain
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed


parser = argparse.ArgumentParser(
    description="Generate augmentation using paraphrases."
)
parser.add_argument("--dataset_name", type=str, default="hugo/knowledge-injection-1")
parser.add_argument("--dataset_config", type=str, default="default")
parser.add_argument("--repeat", type=int, default=1)
parser.add_argument('--batch_size', type=int, required=False, default=8)
parser.add_argument('--temperature', type=float, required=False, default=0.3)
parser.add_argument('--model', type=str, required=False, default='gpt-4o')
parser.add_argument('--api_key', type=str, required=False, default=None)
parser.add_argument('--base_url', type=str, required=False, default="https://api.openai.com/v1")
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

if args.repeat > 1 and args.temperature == 0.0:
    raise ValueError("Repeat > 1 requires a non-zero temperature.")

client = openai.OpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL", args.base_url),
    api_key=os.environ.get("OPENAI_API_KEY", args.api_key),
)

dataset = load_dataset(args.dataset_name, args.dataset_config)["train"]
texts = [example["text"] for example in dataset]

prompt_template = '''Your task is to rewrite the given text while maintaining its original meaning. Do not change any factual details, but focus on rephrasing to convey the same ideas. Aim to paraphrase the content while keeping it approximately the same length as the original. Return only the rewritten text, without preambles such as "Here is the...".
Input text:

{document}'''

def process_example(client, example, max_retries=30, retry_delay=3):
    augmentations = []
    for repeat in range(args.repeat):
        attempt = 0
        while attempt < max_retries:
            try:
                prompt = prompt_template.format(
                    document=example['text'],
                )
                messages = [{"role": "user", "content": prompt}]
                response = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    temperature=args.temperature,
                )
                augmentations.append(response.choices[0].message.content.strip())
                break
            except Exception as e:
                attempt += 1
                print(f"Attempt {attempt} failed with error: {e}. Retrying...")
                import traceback
                traceback.print_exc()
                time.sleep(retry_delay)
    return augmentations


all_augmentations = []
with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
    future_to_example = {
        executor.submit(process_example, client, example): example
        for example in dataset
    }

    for future in tqdm(as_completed(future_to_example), total=len(dataset), desc='Augmenting'):
        results = future.result()
        all_augmentations += results

with open(args.output, 'w') as out:
    for text in chain(texts, all_augmentations):
        out.write(json.dumps({"text": text}) + '\n')
