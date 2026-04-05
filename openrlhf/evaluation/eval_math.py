import argparse
import openrlhf.evaluation.datasets_loader as datasets_loader
from transformers import AutoTokenizer
import json
import os
import re
from vllm import LLM, SamplingParams
import ray

class VLLMWorker:
    def __init__(self, model_path, tensor_parallel_size):

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.85,
        )

    def generate(self, prompts, temperature=0.0, top_p=0.95, max_tokens=4096):
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=True)
        return [out.outputs[0].text for out in outputs]


def check_scores(responses, answers, scores):
    def extract_all_boxed_expressions(text):
        results = []
        pattern = r'\\boxed\{'
        i = 0
        while i < len(text):
            match = re.search(pattern, text[i:])
            if not match:
                break
            start = i + match.start()
            brace_level = 0
            j = start + len(r'\boxed{')
            while j < len(text):
                if text[j] == '{':
                    brace_level += 1
                elif text[j] == '}':
                    if brace_level == 0:

                        results.append(
                            text[start + len(r'\boxed{'):j]
                            .replace(" ", "")
                            .replace("\n", "")
                        )
                        i = j + 1
                        break
                    else:
                        brace_level -= 1
                j += 1
            else:

                break
            i = j
        if len(results) == 0:
            return None
        return results

    for idx, score in enumerate(scores):
        if score == 0:
            response = responses[idx]
            answer = answers[idx]
            boxed_answer = extract_all_boxed_expressions(response)
            if boxed_answer:
                if answer == boxed_answer[-1]:
                    scores[idx] = 1
                try:
                    float_boxed = float(boxed_answer[-1])
                    float_answer = float(answer)
                    if float_boxed == float_answer:
                        scores[idx] = 1
                except Exception:
                    pass
    avg_score = sum(scores) / len(scores)
    return scores, avg_score


def build_prompts(questions, tokenizer):
    chats = [
        [
            {
                "role": "user",
                "content": question
                + "\nPlease reason step by step, and put your final answer within \\boxed{}.",
            },
        ]
        for question in questions
    ]

    chat_template = getattr(tokenizer, "chat_template", None)

    if chat_template:
        prompts = [
            tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,

            )
            for chat in chats
        ]
    else:
        prompts = [("A conversation between User and Assistant. "
            "The user asks a question, and the Assistant solves it. "
            "The assistant first thinks about the reasoning process in the mind "
            "and then provides the user with the answer.\n\n"
            f"User: {chat[0]['content']}\n"
            "Assistant:") for chat in chats]
    return prompts


def save_detailed_results(base_dir, dataset_name, results, tag):
    file_path = os.path.join(base_dir, dataset_name, f"results_{tag}.json")
    os.makedirs(os.path.join(base_dir, dataset_name), exist_ok=True)
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


# NEW: TODO: translate comment
def save_metrics(
    base_dir,
    model_path,
    dataset_name,
    average_score,
    temperature=None,
    run_idx=None,
    num_runs=None,
    tag=None,
):
    summary = {
        "dataset": dataset_name,
        "model": model_path,

        "average_score": average_score,
        "accuracy_percent": round(average_score * 100, 2),
    }

    if temperature is not None:
        summary["temperature"] = temperature
    if run_idx is not None:
        summary["run_idx"] = run_idx
    if num_runs is not None:
        summary["num_runs"] = num_runs
    if tag is not None:
        summary["tag"] = tag

    os.makedirs(base_dir, exist_ok=True)

    final_results_path = os.path.join(base_dir, "final_results.jsonl")
    with open(final_results_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

def evaluate_single_dataset(actors, tokenizer, dataset_name, args, base_dir,
                            temperature=0.0, run_idx=None):
    label = dataset_name if run_idx is None else f"{dataset_name} (run {run_idx})"
    print(f"\n========== Evaluating dataset: {label} ==========\n")


    suffix = "" if run_idx is None else f"_{run_idx}"
    file_path = os.path.join(base_dir, f"{dataset_name}", f"results_{dataset_name}{suffix}.json")


    if os.path.exists(file_path):
        print(f"[INFO] {file_path} already exists, load average_score and skip generation.")
        with open(file_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        avg = None
        if isinstance(saved, list) and len(saved) > 0 and isinstance(saved[-1], dict):
            if "average_score" in saved[-1]:
                avg = saved[-1]["average_score"]
        if avg is None and isinstance(saved, list):
            scores = [
                item.get("score", 0.0)
                for item in saved
                if isinstance(item, dict) and "score" in item
            ]
            if scores:
                avg = sum(scores) / len(scores)
        if avg is None:
            avg = 0.0
        return avg
    
    handler = datasets_loader.get_dataset_handler(dataset_name, args.name)
    questions, answers = handler.load_data()
    
    prompts = build_prompts(questions, tokenizer)

    num_samples = len(prompts)
    num_actors = len(actors)

    chunk_size = (num_samples + num_actors - 1) // num_actors  
    futures = []
    for i in range(num_actors):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, num_samples)
        if start >= end:
            break
        chunk_prompts = prompts[start:end]
        worker = actors[i]
        futures.append(
            worker.generate.remote(
                chunk_prompts,
                temperature=temperature,
                top_p=0.95,
                max_tokens=args.max_tokens,
            )
        )

    responses = []
    output_chunks = ray.get(futures)
    for chunk in output_chunks:
        responses.extend(chunk)

    scores, average_score, pred_answers, golden_answers = handler.get_score_custom(responses, answers)

    count = 0
    for r in responses:
        if len(r) == 0:
            count += 1
    results = [
        {
            "question": question,
            "answer": answer,
            "response": response,
            "score": score,
            "pred_answer": pred_answer,
            "golden_answer": golden_answer
        }
        for question, answer, response, score, pred_answer, golden_answer in zip(
            questions, answers, responses, scores, pred_answers, golden_answers
        )
    ]
    print(f"Average score on {label}: {average_score}")
    results.append({"average_score": average_score})


    # AIME：results_dataset_{run_idx}.json）
    save_detailed_results(
        base_dir,
        dataset_name,
        results,
        dataset_name if run_idx is None else f"{dataset_name}_{run_idx}",
    )

    save_metrics(
        base_dir,
        args.model_path,
        dataset_name,
        average_score,
        temperature=temperature,
        run_idx=run_idx,
    )
    return average_score


TASKS = ["math", "gsm8k", "amc", "minerva", "olympiad", "aime2024", "aime2025"]

def main(args):
    print("Model:", args.model_path)
    print("Dataset option:", args.dataset)

    if "r-zero" in args.model_path.lower() or args.model_path.split('/')[-2] == "solver":
        task = args.model_path.split('/')[-4]
    else:
        if args.model_path.split('/')[-2] == "ckpt":
            task = args.model_path.split('/')[-3]
        else:
            task = args.model_path.split('/')[-2]
    base_dir = args.output_dir + "/" + task + "/" + args.model_path.split('/')[-1]
    os.makedirs(base_dir, exist_ok=True)

    if args.dataset == "all":
        datasets_to_run = TASKS
    else:
        datasets_to_run = [args.dataset]

    ray.init()
    total_gpus = int(ray.available_resources().get("GPU", 0))
    if total_gpus == 0:
        raise RuntimeError(
            "No GPUs detected by Ray. Please make sure GPUs are visible and Ray is configured correctly."
        )

    gpus_per_actor = args.tensor_parallel_size  
    if gpus_per_actor > total_gpus:
        print(
            f"[WARN] tensor_parallel_size ({gpus_per_actor}) > available GPUs ({total_gpus}), "
            f"set gpus_per_actor={total_gpus}"
        )
        gpus_per_actor = total_gpus

    num_actors = max(1, total_gpus // gpus_per_actor)
    print(
        f"[INFO] total_gpus={total_gpus}, "
        f"gpus_per_actor={gpus_per_actor}, "
        f"num_actors={num_actors}"
    )

    VLLMWorkerRemote = ray.remote(num_gpus=gpus_per_actor)(VLLMWorker)

    actors = [
        VLLMWorkerRemote.remote(args.model_path, gpus_per_actor)
        for _ in range(num_actors)
    ]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    all_acc = {}

    for dataset_name in datasets_to_run:
        if dataset_name in ["aime2024", "aime2025"]:
            if dataset_name in ["aime2024", "aime2025"]:
                temperature = 0.6
            else:
                temperature = 0.0
            run_scores = []
            for run_idx in range(1, 33):
                print(
                    f"\n********** AIME multi-run: {dataset_name}, "
                    f"run {run_idx}/32, temperature=0.6 **********"
                )
                avg_score = evaluate_single_dataset(
                    actors,
                    tokenizer,
                    dataset_name,
                    args,
                    base_dir,
                    temperature=temperature,
                    run_idx=run_idx,
                )
                run_scores.append(avg_score)

            if len(run_scores) > 0:
                mean_score = sum(run_scores) / len(run_scores)
            else:
                mean_score = 0.0

            print(
                f"\n========== AIME {dataset_name} 32-run mean accuracy: "
                f"{mean_score} ==========\n"
            )


            save_metrics(
                base_dir,
                args.model_path,
                dataset_name,
                mean_score,
                temperature=0.6,
                num_runs=len(run_scores),
                tag="mean_over_32_runs",
            )

            all_acc[dataset_name] = mean_score

        else:

            avg_score = evaluate_single_dataset(
                actors,
                tokenizer,
                dataset_name,
                args,
                base_dir,
                temperature=0.0,
                run_idx=None,
            )
            all_acc[dataset_name] = avg_score


    final_results_path = os.path.join(base_dir, "final_results.jsonl")
    with open(final_results_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(all_acc, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--name", type=str, default=None)  
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/eval_outputs",
        help="File to save results",
    )
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    args = parser.parse_args()
    main(args)
    
    if "base" not in args.model_path.lower():
        args.max_tokens = 16384
    else:
        args.max_tokens = 8192

    import sys
    sys.exit(0)
