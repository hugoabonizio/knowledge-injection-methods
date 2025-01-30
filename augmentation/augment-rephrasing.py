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
    description="Generate augmentation using Rephrasing the Web technique."
)
parser.add_argument("--dataset_name", type=str, default="hugo/knowledge-injection-1")
parser.add_argument("--dataset_config", type=str, default="default")
parser.add_argument("--repeat", type=int, default=1)
parser.add_argument('--batch_size', type=int, required=False, default=8)
parser.add_argument('--prompts', type=int, required=False, default=4)
parser.add_argument('--temperature', type=float, required=False, default=0.3)
parser.add_argument('--model', type=str, required=False, default='gpt-4o')
parser.add_argument('--api_key', type=str, required=False, default=None)
parser.add_argument('--base_url', type=str, required=False, default="https://api.openai.com/v1")
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

if args.repeat > 1 and args.temperature == 0.0:
    raise ValueError("Repeat > 1 requires a non-zero temperature.")

if args.prompts not in [1, 3, 4]:
    raise ValueError("Prompts must be 1, 3 or 4.")

client = openai.OpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL", args.base_url),
    api_key=os.environ.get("OPENAI_API_KEY", args.api_key),
)

dataset = load_dataset(args.dataset_name, args.dataset_config)["train"]
texts = [example["text"] for example in dataset]

prompt_templates = {
    'easy': '''For the following text give me a paraphrase of the same using a very small vocabulary and extremely simple sentences that a toddler will understand:

{document}''',
    'medium': '''For the following text give me a diverse paraphrase of the same
in high quality English language as in sentences on Wikipedia:

{document}''',
    'hard': '''For the following text give me a paraphrase of the same using very terse and abstruse language that only an erudite scholar will understand. Replace simple words and phrases with rare and complex ones:

{document}''',
    'qa': '''Convert the following text into a conversational format with multiple tags of "Question:" followed by "Answer:":

{document}''',
}

if args.prompts == 1:
    del prompt_templates["easy"]
    del prompt_templates["medium"]
    del prompt_templates["hard"]
elif args.prompts == 3:
    del prompt_templates["qa"]

def process_example(client, example, max_retries=30, retry_delay=3):
    augmentations = []
    for repeat in range(args.repeat):
        attempt = 0
        while attempt < max_retries:
            try:
                for (prompt_type, prompt_template) in prompt_templates.items():
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
                print(f"Attempt {attempt} failed with error: {e}. Retrying...", flush=True)
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
