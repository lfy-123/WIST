import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm

DEFAULT_JUDGE_MODEL = os.environ.get("WIST_RESULTS_RECHECK_MODEL", "gpt-4.1")
DEFAULT_JUDGE_BASE_URL = os.environ.get("WIST_RESULTS_RECHECK_BASE_URL")
DEFAULT_JUDGE_API_KEY = os.environ.get("WIST_RESULTS_RECHECK_API_KEY")

EQUALITY_TEMPLATE = r"""
Look at the following two expressions (answers to a math problem) and judge whether they are equivalent. Only perform trivial simplifications.

Examples:

    Expression 1: $2x+3$
    Expression 2: $3+2x$

Yes

    Expression 1: 3/2
    Expression 2: 1.5

Yes

    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$

No

    Expression 1: $x^2+2x+1$
    Expression 2: $(x+1)^2$

Yes

    Expression 1: 3245/5
    Expression 2: 649

No

    Expression 1: 2/(-3)
    Expression 2: -2/3

Yes

    Expression 1: 72 degrees
    Expression 2: 72

Yes

    Expression 1: 64
    Expression 2: 64 square feet

Yes

---

YOUR TASK

Respond with only "Yes" or "No".

    Expression 1: %(expression1)s
    Expression 2: %(expression2)s
""".strip()


def build_client() -> OpenAI:
    if not DEFAULT_JUDGE_API_KEY or not DEFAULT_JUDGE_BASE_URL:
        raise RuntimeError(
            "Please set WIST_RESULTS_RECHECK_API_KEY and WIST_RESULTS_RECHECK_BASE_URL before running results_recheck.py"
        )
    return OpenAI(api_key=DEFAULT_JUDGE_API_KEY, base_url=DEFAULT_JUDGE_BASE_URL)


def judge_equivalence(answer: str, response: str, client: OpenAI, model_name: str) -> str:
    prompt = EQUALITY_TEMPLATE % {"expression1": answer, "expression2": response}
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a math answer checker."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=32,
        stream=False,
    )
    return completion.choices[0].message.content


def load_results_file(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        raw_results = json.load(f)
    return [
        item
        for item in raw_results
        if isinstance(item, dict) and "question" in item and "answer" in item and "response" in item
    ]


def evaluate_single_file(file_path: str, client: OpenAI, model_name: str) -> Optional[float]:
    if not os.path.exists(file_path):
        print(f"[WARN] Missing results file: {file_path}")
        return None

    items = load_results_file(file_path)
    if not items:
        print(f"[WARN] No valid items found in {file_path}")
        return None

    print(f"Evaluating {len(items)} items from {file_path}")

    to_judge = []
    keep_scores = []
    for idx, item in enumerate(items):
        pred_answer = item.get("pred_answer")
        score = float(item.get("score", 0.0))
        if pred_answer is None:
            keep_scores.append(0)
            continue
        if score < 0.5:
            to_judge.append((idx, item))
        else:
            keep_scores.append(1)

    judged_scores = [0] * len(to_judge)

    def judge_item(item: Dict) -> str:
        return judge_equivalence(item["answer"], item["response"], client, model_name)

    if to_judge:
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = {executor.submit(judge_item, item): idx for idx, (_, item) in enumerate(to_judge)}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Judging items"):
                position = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"[WARN] Judge request failed for item {position}: {exc}")
                    result = "No"
                judged_scores[position] = 1 if "yes" in str(result).lower() else 0

    final_scores = keep_scores + judged_scores
    accuracy = sum(final_scores) / len(final_scores)
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    return accuracy


def main(args):
    client = build_client()
    acc = evaluate_single_file(args.input_file, client, args.judge_model)
    if acc is not None:
        base_dir = os.path.dirname(args.input_file)
        final_results_path = os.path.join(base_dir, "final_results.jsonl")
        with open(final_results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({f"Check Model: {args.judge_model}": acc}, ensure_ascii=False) + "\n")
        print("\nJudge-based accuracy:")
        print(f"  {args.input_file}: {acc:.4f} ({acc * 100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to the results_xxx.json file")
    parser.add_argument("--judge_model", type=str, default=DEFAULT_JUDGE_MODEL)
    main(parser.parse_args())
