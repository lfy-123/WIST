import datasets
import json
import re
import random
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import os
# from openrlhf.trainer.ppo_utils.utils import preprocess_data

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


def extract_last_boxed(text):
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
    matches = list(re.finditer(pattern, text))
    if matches:
        return matches[-1].group(1)
    return None

def extract_last_final_answer(text):
    pattern1 = r'Final Answer:((?:[^<]|<[^<])*?)\n'
    pattern2 = r'The answer is:((?:[^<]|<[^<])*?)\n'
    pattern3 = r'answer is:((?:[^<]|<[^<])*?)\n'
    matches1 = list(re.finditer(pattern1, text))
    matches2 = list(re.finditer(pattern2, text))
    pattern3 = list(re.finditer(pattern3, text))
    if matches1:
        return matches1[-1].group(1)
    elif matches2:
        return matches2[-1].group(1)
    elif pattern3:
        return pattern3[-1].group(1)
    return None

def extract_solution(solution_str):
    if '<|im_start|>user' in solution_str:
        model_output = re.sub(r'^.*?<\|im_start\|>assistant', '<|im_start|>assistant', solution_str, flags=re.DOTALL, count=1)
    elif 'Assistant:' in solution_str:
        model_output = solution_str.split('Assistant:')[-1].strip()
    else:
        model_output = solution_str
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    for stop_word in stop_words:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()
    extract_boxed_answer = extract_last_boxed(model_output)
    if extract_boxed_answer:
        return extract_boxed_answer
    else:
        return extract_last_final_answer(model_output)

def form_options(options: list):
    option_str = 'Options are:\n'
    opts = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    for opt, o in zip(options, opts):
        option_str += f'({o}): {opt}\n'
    return option_str

def get_prediction(output):
    solution = extract_solution(output)
    if solution is None:
        # return random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
        return ""
    solu_answer = []
    for option in ['A', 'B', 'C', 'D']:
        if option in solution:
            # return option
            solu_answer.append(option)
    if len(solu_answer) == 1:
        return solu_answer[0]
    else:
    # return random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
        return ""






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--output_dir", type=str, default="./outputs/eval_outputs", help="File to save results")
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    args = parser.parse_args()
    
    if "base" not in args.model_path.lower():
        max_tokens = 16384
    else:
        max_tokens = 8192

    if "r-zero" in args.model_path.lower() or args.model_path.split('/')[-2] == "solver":
        task = args.model_path.split('/')[-4]
    else:
        if args.model_path.split('/')[-2] == "ckpt":
            task = args.model_path.split('/')[-3]
        else:
            task = args.model_path.split('/')[-2]
    output_ckpt = args.output_dir + "/" + task + "/" + args.model_path.split('/')[-1]
    os.makedirs(output_ckpt, exist_ok=True)

    output_file = os.path.join(output_ckpt, "gpqa_diamond.json")
    
    ray.init()
    total_gpus = int(ray.available_resources().get("GPU", 0))
    if total_gpus == 0:
        raise RuntimeError("No GPUs detected by Ray. Please make sure GPUs are visible and Ray is configured correctly.")

    gpus_per_actor = args.tensor_parallel_size
    if gpus_per_actor > total_gpus:
        print(f"[WARN] tensor_parallel_size ({gpus_per_actor}) > available GPUs ({total_gpus}), set gpus_per_actor={total_gpus}")
        gpus_per_actor = total_gpus

    num_actors = max(1, total_gpus // gpus_per_actor)
    print(f"[INFO] total_gpus={total_gpus}, gpus_per_actor={gpus_per_actor}, num_actors={num_actors}")
    VLLMWorkerRemote = ray.remote(num_gpus=gpus_per_actor)(VLLMWorker)

    actors = [
        VLLMWorkerRemote.remote(args.model_path, gpus_per_actor)
        for _ in range(num_actors)
    ]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    dataset = datasets.load_dataset('/inspire/hdd/global_user/lifangyuan-253108110077/lifangyuan/self_improve/R-Zero/data/GPQA-Diamond')["test"]
    
    prompts = []
    for entry in dataset:
        query = entry['question']
        messages = [{
            "role": "user",
            "content": query + '\nPlease reason step by step, and put your final answer option within \\boxed{}. Only put the option letter in the box, e.g. \\boxed{A}. There is only one correct answer.'
        }]
        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # , enable_thinking=False
        else:
            prompt = ("A conversation between User and Assistant. "
                "The user asks a question, and the Assistant solves it. "
                "The assistant first thinks about the reasoning process in the mind "
                "and then provides the user with the answer.\n\n"
                f"User: {query}\n"
                "Please reason step by step, and put your final answer within \\boxed{}. Only put the letter in the box, e.g. \\boxed{A}. There is only one correct answer.\n\n"
                "Assistant:")
                            
        prompts.append(prompt)
    print(f"================= Constructed prompt example: {prompts[0]}")

    num_samples = len(prompts)
    assert num_samples != 0, "[ERROR] No samples found in the dataset!"

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
                temperature=0.0,
                top_p=1.0,
                max_tokens=max_tokens,
            )
        )

    outputs = []
    output_chunks = ray.get(futures)
    for chunk in output_chunks:
        outputs.extend(chunk)
    
    success = 0
    fail = 0
    answers = []
    for entry, answer in zip(dataset, outputs):
        prediction = get_prediction(answer)
        entry['solution'] = answer
        entry['predicted_answer'] = prediction
        if entry["answer"] == prediction:
            success += 1
            entry['judge_result'] = "success"
        else:
            fail += 1
            entry['judge_result'] = "fail"
        answers.append(entry)
        
    accuracy = success / (success + fail)
    # Print category accuracy as soon as it's computed
    print(f"GPQA-Diamond: {accuracy:.4f}")
    
    with open(output_file, 'w') as f:
        json.dump(answers, f, indent=2)
    
    summary = {
        "dataset": "GPQA_Diamond",
        "accuracy": round(accuracy * 100, 2),
        "model": args.model_path,
    }
    
    with open(os.path.join(output_ckpt, "final_results.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    
    import sys
    sys.exit(0)
    
    