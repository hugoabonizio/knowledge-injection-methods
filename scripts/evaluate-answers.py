import os
import re
import time
import json
import openai
import argparse
import textwrap
import pandas as pd
from tqdm import tqdm
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed


parser = argparse.ArgumentParser(description="Evaluate model's answers.")
parser.add_argument(
    "--generations",
    type=str,
    required=True,
    help="Path to the generations JSON file.",
)
parser.add_argument(
    "--output",
    type=str,
    required=True,
    help="Path to the output JSON file with judged answers.",
)
parser.add_argument(
    "--api_key",
    type=str,
    required=True,
    help="API key.",
)
parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1")
parser.add_argument("--judge", type=str, default="gpt-4o-2024-08-06")
parser.add_argument("--max_tokens", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--json", action="store_true", help="Format outputs as JSON")
args = parser.parse_args()


def extract_prediction(response):
    match = re.search(r"Correct:\s*(yes|no)", response, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    else:
        return None


def process_example(client, prompt_template, example, max_retries=30, retry_delay=2):
    attempt = 0
    while attempt < max_retries:
        try:
            prompt = prompt_template.format(
                question=example["question"],
                expected_answer=example["answer_true"],
                model_answer=example["answer_model"],
            )
            messages = [{"role": "user", "content": prompt}]

            response = client.chat.completions.create(
                model=args.judge,
                messages=messages,
                temperature=0.0,
                max_tokens=args.max_tokens,
            )
            answer = response.choices[0].message.content
            result = extract_prediction(answer)
            return {**example, "result": result}
        except Exception as e:
            attempt += 1
            print(f"Attempt {attempt} failed with error: {e}. Retrying...")
            time.sleep(retry_delay)
    print(f"Failed to process example after {max_retries} attempts.")
    return None


client = openai.OpenAI(
    base_url=args.base_url,
    api_key=args.api_key,
)

prompt_template = """I will provide a question, an expected answer and the candidate answer. Your task is to verify if the candidate answer is correct. The expected answer is the ground-truth, so if the candidate answer contradicts the expected answer or refuses to answer, it is incorrect.

Question: "{question}"
Expected answer: "{expected_answer}"
Candidate answer: "{model_answer}"

Answer in the format
Reasoning: (your reasoning)
Correct: (yes|no)"""

results = []

if os.path.exists(args.output):
    with open(args.output) as f:
        for line in f:
            results.append(json.loads(line))
else:
    with open(args.generations) as f:
        examples = [json.loads(line) for line in f]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as out:
        with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
            future_to_example = {
                executor.submit(
                    process_example, client, prompt_template, example
                ): example
                for example in examples
            }

            for future in tqdm(
                as_completed(future_to_example), total=len(examples), desc="Evaluating"
            ):
                example_results = future.result()
                results.append(example_results)
                out.write(json.dumps(example_results) + "\n")

df = pd.DataFrame(results)
df["year"] = df["year"].astype(int)
df_old = df.query("year <= 2022")
correct_old = sum(df_old["result"].str.match("yes"))
df_new = df.query("year > 2022")
correct_new = sum(df_new["result"].str.match("yes"))

if args.json:
    print(
        json.dumps(
            {
                "accuracy_old": (correct_old / len(df_old) * 100),
                "accuracy_new": (correct_new / len(df_new) * 100),
            }
        )
    )
else:
    print(f"Accuracy (old): {(correct_old / len(df_old) * 100):.1f}")
    print(f"Accuracy (new): {(correct_new / len(df_new) * 100):.1f}")
