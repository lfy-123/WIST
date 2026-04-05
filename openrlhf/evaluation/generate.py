import vllm
import argparse
import evaluation.datasets_loader as datasets_loader
from transformers import AutoTokenizer
import json
import os
import re
STORAGE_PATH = os.getenv("STORAGE_PATH")

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
                        results.append(text[start + len(r'\boxed{'):j].replace(" ", "").replace("\n", ""))
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
                except:
                    pass
    avg_score = sum(scores)/len(scores)
    return scores, avg_score


def main(args):
    print("STORAGE_PATH")
    print(STORAGE_PATH)
    with open('tokens.json','r') as f: 
        tokens = json.load(f)
    print(args.model, args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = vllm.LLM(
        model=args.model,
        tokenizer=args.model,
        gpu_memory_utilization=0.85
    )
    sample_params = vllm.SamplingParams(
        max_tokens=4096,
        temperature=0.0,
        stop_token_ids=[tokenizer.eos_token_id],
    )
    handler = datasets_loader.get_dataset_handler(args.dataset,args.name)
    questions, answers = handler.load_data()
    chats=[[{"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},{"role": "user", "content": question}] for question in questions]
    if tokenizer.chat_template:
        prompts = [tokenizer.apply_chat_template(chat, tokenize=False,add_generation_prompt=True, add_special_tokens=True, enable_thinking=False) for chat in chats]
    else:
        prompts = ["system: " + chat[0]["content"] + '\n' + "user: " + chat[1]["content"] + '\nPlease reason step by step, and put your final answer within \\boxed{}.' for chat in chats]
    responses = model.generate(prompts, sampling_params=sample_params,use_tqdm=True)
    responses = [response.outputs[0].text for response in responses]
    scores,average_score = handler.get_score(responses, answers)

    results = [{"question": question, "answer": answer, "response": response, "score": score} for question, answer, response, score in zip(questions, answers, responses, scores)]
    print(f"Average score: {average_score}")
    results.append({"average_score": average_score})

    model_name = args.model.split('/')[-4] if args.model.split('/')[-1] == "huggingface" else args.model.split('/')[-1]
    expert_name = args.model.split('/')[-5]
    os.makedirs(f"{STORAGE_PATH}/evaluation/{expert_name}/{model_name}", exist_ok=True)
    with open(f"{STORAGE_PATH}/evaluation/{expert_name}/{model_name}/results_{args.dataset}.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--dataset", type=str, default="math")
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()
    main(args)