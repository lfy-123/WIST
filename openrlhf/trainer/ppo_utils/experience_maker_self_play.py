import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass, fields
from datetime import timedelta
from typing import Any, List, Tuple, Union, Dict, Optional
from math_verify import parse, verify
import numpy as np
import ray
import torch
import unicodedata

from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.seqlen_balancing import get_minimum_num_micro_batch_size, get_seqlen_balanced_partitions
from openrlhf.utils.utils import remove_pad_token, zero_pad_sequences
from openrlhf.prompts.utils import  parse_json_output, parse_box_output, extract_all_boxed_expressions, extract_mcq_options, normalize_pred_answers
from openrlhf.prompts.reasoner import REASONER_PROMPT
from vllm import SamplingParams
import re
from openrlhf.prompts.judge import QUESTION_TYPE_JUDGE_PROMPT
from openrlhf.prompts.challenger import MCQ_PROMPT_WITH_PATH_MATH, FREE_PROMPT_WITH_PATH_MATH
from openrlhf.prompts.challenger import MCQ_PROMPT_WITH_PATH_MEDICINE, FREE_PROMPT_WITH_PATH_MEDICINE
from openrlhf.prompts.challenger import MCQ_PROMPT_WITH_PATH_PHYSICS, FREE_PROMPT_WITH_PATH_PHYSICS

import random
from openrlhf.trainer.ppo_utils.utils import is_valid_free_qa_from_model_output, preprocess_data, preprocess_challenger_data
import functools
from openrlhf.trainer.ppo_utils.math_grader import boxed_reward_fn

logger = init_logger(__name__)

def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor

def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor

@dataclass
class Experience:
    """Experience is a batch of data for RLHF training.

    Shapes of each tensor:
    index: (B,)
    sequences: (B, S)
    attention_mask: (B, S)
    action_mask: (B, A)
    action_log_probs: (B, S)
    base_action_log_probs: (B, S)
    values: (B, S)
    returns: (B, S)
    advantages: (B, S)
    kl: (B, S)
    info: dict[str, list]
    """

    index: list[int] = None
    sequences: torch.Tensor = None
    attention_mask: torch.LongTensor = None
    action_mask: torch.BoolTensor = None

    action_log_probs: torch.Tensor = None
    base_action_log_probs: torch.Tensor = None
    rollout_log_probs: torch.Tensor = None
    values: torch.Tensor = None
    returns: torch.Tensor = None
    advantages: torch.Tensor = None
    kl: torch.Tensor = None

    prompts: list[str] = None
    labels: list[str] = None
    rewards: torch.Tensor = None  # used for advantage calculation
    scores: torch.Tensor = None  # 0-1 reward used for dynamic sampling

    # the info field is used to store additional information
    # all the fields in the info will be logged to the tensorboard/wandb
    info: dict[str, torch.Tensor] = None
    
    questions: list[str] = None
    answers: list[str] = None
    question_type: list[str] = None  # MCQ or free-form

    def __init__(
        self,
        index=None,
        sequences=None,
        action_log_probs=None,
        base_action_log_probs=None,
        rollout_log_probs=None,
        values=None,
        returns=None,
        advantages=None,
        attention_mask=None,
        action_mask=None,
        kl=None,
        prompts=None,
        labels=None,
        rewards=None,
        scores=None,
        info=None,
        questions=None,
        answers=None,
        question_type=None,
    ):
        self.index = index
        self.sequences = sequences
        self.action_log_probs = action_log_probs
        self.base_action_log_probs = base_action_log_probs
        self.rollout_log_probs = rollout_log_probs
        self.values = values
        self.returns = returns
        self.advantages = advantages
        self.attention_mask = attention_mask
        self.action_mask = action_mask
        self.kl = kl
        self.prompts = prompts or []
        self.labels = labels or []
        self.rewards = rewards
        self.scores = scores
        self.info = info or []
        
        self.questions = questions or []
        self.answers = answers or []
        self.question_type = question_type or []
        
    @torch.no_grad()
    def to_device(self, device: torch.device):
        """Move all tensor fields to the specified device."""
        for field, value in self.__dict__.items():
            if isinstance(value, dict):
                setattr(self, field, {key: to(val, device) for key, val in value.items()})
            else:
                setattr(self, field, to(value, device))

        return self

    def pin_memory(self):
        """Pin memory for all tensor fields."""
        for field, value in self.__dict__.items():
            if isinstance(value, dict):
                setattr(self, field, {key: pin_memory(val) for key, val in value.items()})
            else:
                setattr(self, field, pin_memory(value))

        return self

    @staticmethod
    def select(experiences: List["Experience"], fields: List[str]) -> List["Experience"]:
        """Select specific fields from a list of Experience instances to create new Experience instances.

        Args:
            experiences: List of Experience instances
            fields: List of field names to select

        Returns:
            A list of new Experience instances containing only the selected fields
        """
        new_experiences = []
        for exp in experiences:
            new_exp = Experience()
            for field in fields:
                if hasattr(exp, field):
                    setattr(new_exp, field, getattr(exp, field))
            new_experiences.append(new_exp)
        return new_experiences

    @staticmethod
    def _merge_item(items: List, pad_value: int = 0) -> Union[torch.Tensor, list, dict, Any]:
        """Merge a list of items into a single item.
        Recursively merge tensors, lists and dicts.
        For tensors, use zero_pad_sequences to merge sequences of different lengths.

        Args:
            items: List of items to merge
            pad_value: Value used for padding tensors
        """
        if isinstance(items[0], torch.Tensor):
            return zero_pad_sequences(items, side="right", value=pad_value)
        elif isinstance(items[0], list):
            return sum(items, [])
        elif isinstance(items[0], dict):
            result = {}
            # Collect all values for each key
            for d in items:
                for key, value in d.items():
                    if key not in result:
                        result[key] = []
                    result[key].append(value)
            # Merge all values for each key at once
            return {key: Experience._merge_item(values, pad_value) for key, values in result.items()}
        elif items[0] is None:
            return None
        else:
            raise ValueError(f"Unsupported type: {type(items[0])}")

    @staticmethod
    def concat_experiences(experiences_list: List["Experience"], pad_token_id) -> "Experience":
        """Concatenate multiple experiences into one large experience.

        Args:
            experiences_list: List of Experience to concatenate
            pad_token_id: Token id used for padding sequences

        Returns:
            A new Experience instance containing all the concatenated data
        """
        if not experiences_list:
            return Experience()

        # Get all field names from the dataclass
        field_names = [f.name for f in fields(Experience)]

        # Create result dictionary
        result = {}

        # Merge all fields
        for field in field_names:
            values = [getattr(e, field) for e in experiences_list]
            # Use pad_token_id for sequences field, 0 for others
            pad_value = pad_token_id if field == "sequences" else 0
            result[field] = Experience._merge_item(values, pad_value)

        return Experience(**result)


def update_samples_with_rewards(rewards_info, samples_list):
    """Process rewards info and update samples with rewards, scores and extra logs.

    Args:
        rewards_info: List of reward information dictionaries
        samples_list: List of Experience objects to update
    """
    # Process rewards and scores
    samples_len = [len(sample.sequences) for sample in samples_list]

    rewards_list = torch.cat([torch.as_tensor(info["rewards"]) for info in rewards_info], dim=0).split(samples_len)
    if "scores" in rewards_info[0]:
        scores_list = torch.cat([torch.as_tensor(info["scores"]) for info in rewards_info], dim=0).split(samples_len)
    else:
        scores_list = rewards_list

    # Process extra_logs if present
    if "extra_logs" in rewards_info[0]:
        # Merge all extra_logs tensors first
        merged_logs = {
            key: torch.cat(
                [torch.as_tensor(logs[key]) for logs in [info["extra_logs"] for info in rewards_info]], dim=0
            ).split(samples_len)
            for key in rewards_info[0]["extra_logs"].keys()
        }

    # Update samples with rewards, scores and extra logs
    for i, samples in enumerate(samples_list):
        samples.rewards = rewards_list[i]
        samples.scores = scores_list[i]
        samples.info["score"] = scores_list[i]
        samples.info["reward"] = rewards_list[i]
        if "extra_logs" in rewards_info[0]:
            for key, values in merged_logs.items():
                samples.info[key] = values[i]

    return samples_list

def is_mcq_question_text(text: str, min_question_len: int = 10) -> bool:
    """
    Determine if a given question text is a "decent" four-choice multiple-choice question.
    Rules:
    1. The text must contain the four option markers (A)/B)/C)/D) or A./B./C./D.);
    2. After extracting the stem (the content before A), removing spaces and punctuation, the length must be >= min_question_len.
    """
    if not isinstance(text, str) or not text.strip():
        return False

    option_pattern = re.compile(r'(?:^|\s)([A-D])[\.\)]\s+', re.MULTILINE)
    labels = {m.group(1) for m in option_pattern.finditer(text)}

    if not all(ch in labels for ch in "ABCD"):
        return False

    m_a = re.search(r'\bA[\.\)]', text)
    if not m_a:
        return False

    stem = text[:m_a.start()].strip()

    stem = re.sub(r'^\s*(Question|Q)\s*:\s*', '', stem, flags=re.IGNORECASE).strip()

    stem_core = re.sub(r'[\.\?？!！…\s]', '', stem)

    if len(stem_core) < min_question_len:
        return False
    return True

def is_valid_mcq_answer_letter(ans: str) -> bool:
    """
    Verify if a single answer is a valid multiple-choice option:
    - Must be one of 'A' / 'B' / 'C' / 'D' (case insensitive)
    - 'AB', 'A ', '1', '' and similar cases are not allowed
    """
    if not isinstance(ans, str):
        return False
    s = ans.strip()
    return s in {"A", "B", "C", "D"} and len(s) == 1

def has_abcd_options(text: str) -> bool:
    """
    Check if the text contains options similar to 'A) xxx' / 'B. yyy' / 'C) zzz' / 'D. www'. 
    This is only used to exclude the situation where free-response questions are generated as multiple-choice questions.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    pattern = re.compile(r'(?:^|\s)([A-D])[\.\)]\s+', re.MULTILINE)
    labels = {m.group(1) for m in pattern.finditer(text)}
    return all(ch in labels for ch in "ABCD")


def is_valid_answer_by_type(answer: str, answer_type: str) -> bool:
    """
    Check the basic form of correct_answer according to answer_type.
    Here, only a "form" check is performed, not a mathematical correctness verification.
    """
    if not isinstance(answer, str):
        return False
    ans = answer.strip()
    if not ans:
        return False
    if answer_type == "integer":
        return re.fullmatch(r"[+-]?\d+", ans) is not None
    if answer_type == "real_number":
        return re.fullmatch(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?", ans) is not None
    if answer_type == "expression":
        core = re.sub(r'\s', '', ans)
        return len(core) >= 2  
    if answer_type == "string":
        core = re.sub(r'[.\·•…\s]', '', ans)
        return len(core) >= 1
    return False

class SamplesGenerator:
    def __init__(self, vllm_engines, strategy, tokenizer, prompt_max_len):
        self.strategy = strategy
        self.args = strategy.args
        self.vllm_engines = vllm_engines
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.challenger_question_types = None
        self.challenger_output_log_path = None
        
        self.invalid_penalty = self.args.format_invalid_penalty if hasattr(self.args, "format_invalid_penalty") else -1.0
        self.judge_question_type = False
        self.apply_chat_template = getattr(tokenizer, 'apply_chat_template', None)
        
        self.generate_qa_prompt = self.args.generate_qa_prompt if hasattr(self.args, "generate_qa_prompt") else "math"
        
        self.math_reward_fn = functools.partial(
            boxed_reward_fn,
            fast=False,
            correct_reward=1.0,
            incorrect_reward=0.0,
        )
                
    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], all_labels, role: str = None, paths = None, cold_start =False, return_reasoner_experiences: bool = False, challenger_output_log_path = None, **generate_kwargs) -> List[Experience]:
        """
        Generate samples and return in batches.

        When not using vllm, we will fallback to the default implementation,
        in which actor will be used to generate samples.
        """
        self.challenger_output_log_path = challenger_output_log_path
        # vLLM wakeup when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        if cold_start:
            rollout_samples = self._generate_vllm_cold_start(all_prompts, all_labels, role, paths=paths, **generate_kwargs)
        else:
            rollout_samples = self._generate_vllm(all_prompts, all_labels, role, paths=paths, return_reasoner_experiences=return_reasoner_experiences, **generate_kwargs)

        # vLLM offload when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

        return rollout_samples

    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    def get_question_type(self, all_prompts: List[str], sampling_params):
        llms = self.vllm_engines
        judge_prompts = []
        question_types = []
        success_count = 0
        for content in all_prompts:
            judge_prompts.append(QUESTION_TYPE_JUDGE_PROMPT.replace("{text}", content))
        all_prompt_token_ids = self.tokenize_fn(judge_prompts, self.prompt_max_len, padding=False)["input_ids"]

        # Distribute requests to engines and collect responses
        refs = []
        batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]
            refs.append(llm.add_requests.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids))
        ray.get(refs)

        # Retrieve and combine results from all outputs
        all_output_refs = []
        for i, llm in enumerate(llms):
            all_output_refs.append(llm.get_responses.remote())
        all_outputs = sum(ray.get(all_output_refs), [])

        for i in range(len(all_outputs)):
            output = all_outputs[i]
            raw_text = output.outputs[0].text.strip()
            output_dict, test_output = parse_json_output(raw_text)
            if output_dict is None:
                question_types.append(random.choice(["mcq_with_random", "free_with_random"]))
                continue
            suitable_for_mcq = str(output_dict.get("suitable_for_mcq", "false"))
            suitable_for_free_form = str(output_dict.get("suitable_for_free_form", "false"))
            if suitable_for_mcq.lower() == "true" and suitable_for_free_form.lower() == "false":
                question_types.append("mcq")
                success_count += 1
            elif suitable_for_mcq.lower() == "false" and suitable_for_free_form.lower() == "true":
                question_types.append("free")
                success_count += 1
            else:  
                question_types.append(random.choice(["mcq_with_random", "free_with_random"]))
        logger.info(f"Question type judge success rate: {success_count}/{len(all_outputs)}")
        return question_types

    def _generate_vllm(self, all_prompts: List[str], all_labels, role: str, paths: List[str], **kwargs) -> List[Experience]:
        """Generate samples using vLLM engine.
        Args:
            all_prompts: List of prompts to generate from
            all_labels: List of labels corresponding to prompts
            role: Role for experience making (e.g., "challenger", "reasoner")
            **kwargs: Additional arguments for generation

        Returns:
            List of Experience objects containing generated samples
        """
        llms = self.vllm_engines
        args = self.strategy.args
        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
            logprobs=1 if self.strategy.args.enable_vllm_is_correction else None,
        )
        # max_response_length = kwargs.get("max_new_tokens", 1024)
        # truncate_length = self.prompt_max_len + max_response_length

        # Expand prompt list based on the number of samples per prompt
        n_samples_per_prompt = kwargs.pop("n_samples_per_prompt", args.n_samples_per_prompt)
        return_reasoner_experiences = bool(kwargs.pop("return_reasoner_experiences", False))        
        return_generation_logs = bool(kwargs.pop("return_generation_logs", False))

        num_reasoner_samples = int(kwargs.get("num_reasoner_samples", n_samples_per_prompt))
        if num_reasoner_samples != n_samples_per_prompt:
            raise ValueError(
                f"Require num_reasoner_samples == n_samples_per_prompt for balanced experiences: "
                f"{num_reasoner_samples} vs {n_samples_per_prompt}"
            )
        reasoner_max_new_tokens = int(kwargs.get("reasoner_max_new_tokens", kwargs.get("max_new_tokens", 1024)))
        max_retry = int(kwargs.get("challenger_max_resample_attempts", getattr(args, "challenger_max_resample_attempts", 3)))

        def _run_vllm_and_build_samples(
            cur_prompts: List[str],
            cur_labels: List[Any],
            cur_prompt_token_ids: Optional[List[List[int]]] = None,
            sp: Optional[SamplingParams] = None,
            max_new_tokens_local: Optional[int] = None,
        ):
            if not cur_prompts:
                return [], []

            if cur_prompt_token_ids is None:
                all_prompt_token_ids = self.tokenize_fn(cur_prompts, self.prompt_max_len, padding=False)["input_ids"]
            else:
                all_prompt_token_ids = cur_prompt_token_ids
                assert len(all_prompt_token_ids) == len(cur_prompts)

            use_sp = sp or sampling_params
            use_max_new = max_new_tokens_local if max_new_tokens_local is not None else kwargs.get("max_new_tokens", 1024)
            truncate_length = self.prompt_max_len + int(use_max_new)

            refs = []
            batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms)
            for i, llm in enumerate(llms):
                prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]
                if not prompt_token_ids:
                    continue
                refs.append(llm.add_requests.remote(sampling_params=use_sp, prompt_token_ids=prompt_token_ids))
            ray.get(refs)

            all_output_refs = [llm.get_responses.remote() for llm in llms]
            all_outputs = sum(ray.get(all_output_refs), [])

            samples_list: List[Experience] = []
            for i, out in enumerate(all_outputs):
                prompt = cur_prompts[i]
                label = cur_labels[i]

                input_ids = list(out.prompt_token_ids) + list(out.outputs[0].token_ids)
                attention_mask = [1] * len(input_ids)

                sequences = torch.tensor(input_ids, dtype=torch.long)
                attention_mask_t = torch.tensor(attention_mask, dtype=torch.long)

                action_mask = torch.zeros_like(attention_mask_t)
                resp_len = len(out.outputs[0].token_ids)
                action_mask[len(out.prompt_token_ids) : len(out.prompt_token_ids) + resp_len] = 1

                rollout_log_probs = None
                if self.strategy.args.enable_vllm_is_correction:
                    lp = []
                    response_ids = list(out.outputs[0].token_ids)
                    for j, logprob in enumerate(out.outputs[0].logprobs):
                        lp.append(logprob[response_ids[j]].logprob)
                    rollout_log_probs = torch.tensor([0.0] * len(list(out.prompt_token_ids)) + lp)
                    rollout_log_probs = rollout_log_probs[1:truncate_length].to("cpu")

                sequences = sequences[:truncate_length].to("cpu")
                attention_mask_t = attention_mask_t[:truncate_length].to("cpu")
                action_mask = action_mask[1:truncate_length].to("cpu")

                info = {
                    "response_length": torch.tensor([resp_len]),
                    "total_length": torch.tensor([attention_mask_t.float().sum()]),
                    "response_clip_ratio": torch.tensor([resp_len >= int(use_max_new)]),
                }

                exp = Experience(
                    sequences=sequences.unsqueeze(0),
                    attention_mask=attention_mask_t.unsqueeze(0),
                    action_mask=action_mask.unsqueeze(0),
                    rollout_log_probs=rollout_log_probs.unsqueeze(0) if rollout_log_probs is not None else None,
                    prompts=[prompt],
                    labels=[label],
                    info=info,
                )
                samples_list.append(exp)

            return all_outputs, samples_list

        def _run_vllm_outputs_only(prompt_token_ids_list: List[List[int]], sp: SamplingParams):
            refs = []
            batch_size = (len(prompt_token_ids_list) + len(llms) - 1) // len(llms)
            for i, llm in enumerate(llms):
                sub = prompt_token_ids_list[i * batch_size : (i + 1) * batch_size]
                if not sub:
                    continue
                refs.append(llm.add_requests.remote(sampling_params=sp, prompt_token_ids=sub))
            ray.get(refs)
            out_refs = [llm.get_responses.remote() for llm in llms]
            return sum(ray.get(out_refs), [])

        # ---------- challenger reward ----------
        def _compute_challenger_reward(correct_flags: List[float]) -> float:
            if self.args.challenger_rm_mode.lower() == "spice":
                var = np.var(np.asarray(correct_flags, dtype=float))
                return float(np.exp(- (var - 0.25) ** 2 / (2 * 0.01)))
            elif self.args.challenger_rm_mode.lower() == "r-zero":
                p_hat = float(sum(correct_flags)) / max(len(correct_flags), 1)
                return float(1.0 - 2.0 * abs(p_hat - 0.5))
            else:
                raise ValueError(f"Unknown challenger_rm_mode: {self.args.challenger_rm_mode}")

        def _is_extreme(correct_flags: List[float]) -> bool:
            return all(v == 1.0 for v in correct_flags) or all(v == 0.0 for v in correct_flags)

        if role == "challenger":
            original_prompts = all_prompts
            original_paths = paths
            num_original = len(original_prompts)
            generation_logs: List[Dict[str, Any]] = []
            
            if self.judge_question_type:
                question_types_full = self.get_question_type(original_prompts, sampling_params)  # TODO: TODO: translate comment
            else:
                question_types_full = ['mcq'] * num_original  
            assert len(question_types_full) == num_original, "question_types length mismatch"
            
            all_final_prompts: List[str] = []
            for ques_type, content, path in zip(question_types_full, original_prompts, original_paths):
                if self.generate_qa_prompt == "math":
                    if "mcq" in ques_type.lower():
                        p = MCQ_PROMPT_WITH_PATH_MATH.replace("{text}", content).replace("{path}", path)
                    else:
                        p = FREE_PROMPT_WITH_PATH_MATH.replace("{text}", content).replace("{path}", path)
                elif self.generate_qa_prompt == "med":
                    if "mcq" in ques_type.lower():
                        p = MCQ_PROMPT_WITH_PATH_MEDICINE.replace("{text}", content).replace("{path}", path)
                    else:
                        p = FREE_PROMPT_WITH_PATH_MEDICINE.replace("{text}", content).replace("{path}", path)
                elif self.generate_qa_prompt == "phy":
                    if "mcq" in ques_type.lower():
                        p = MCQ_PROMPT_WITH_PATH_PHYSICS.replace("{text}", content).replace("{path}", path)
                    else:
                        p = FREE_PROMPT_WITH_PATH_PHYSICS.replace("{text}", content).replace("{path}", path)
                else:
                    raise ValueError(f"Unknown generate_qa_prompt: {self.generate_qa_prompt}")
                p = preprocess_challenger_data(p, apply_chat_template=self.apply_chat_template, template_type=self.strategy.args.template_type)
                all_final_prompts.append(p)

            all_final_prompt_token_ids: List[List[int]] = self.tokenize_fn(
                all_final_prompts, self.prompt_max_len, padding=False
            )["input_ids"]
            
            

            if any(len(t) >= self.prompt_max_len for t in all_final_prompt_token_ids):
                raise ValueError("Some challenger prompts are too long and get truncated; raise instruction length limit")

            # >>> SPICE MOD START: oversample then subsample-to-G

            challenger_n_candidates_per_prompt = kwargs.get(
                "challenger_n_candidates_per_prompt",
                getattr(args, "challenger_n_candidates_per_prompt", None),
            )
            if challenger_n_candidates_per_prompt is None:
                challenger_n_candidates_per_prompt = n_samples_per_prompt
            # challenger_n_candidates_per_prompt = max(int(challenger_n_candidates_per_prompt), int(n_samples_per_prompt) * 2)
            # >>> SPICE MOD END
            
            # >>> SPICE MOD START: helper for ratio-preserving subsample
            def _subsample_preserve_valid_invalid_ratio(valid_ids: List[int], invalid_ids: List[int], target_k: int) -> List[int]:
                """
                Return a list of global indices of length target_k (when possible),
                preserving valid:invalid ratio, and if both exist, keep at least 1 from each.
                """
                V, I = len(valid_ids), len(invalid_ids)
                M = V + I
                if target_k <= 0 or M == 0:
                    return []
                if target_k >= M:
                    # Already have <= target_k candidates
                    return list(valid_ids) + list(invalid_ids)

                # Prefer valid if only one class exists
                if I == 0:
                    return random.sample(valid_ids, k=min(target_k, V))
                if V == 0:
                    # caller should handle resample; keep empty to indicate failure
                    return []
                if target_k == 1:
                    # Can't keep both classes; prefer valid
                    return [random.choice(valid_ids)]
                # Proportional target counts
                n_valid = int(round(target_k * V / M))
                n_valid = max(1, min(target_k - 1, n_valid))
                n_invalid = target_k - n_valid

                # Availability adjustment
                n_valid = min(n_valid, V)
                n_invalid = min(n_invalid, I)

                # Ensure at least 1 from each when both exist (and target_k>=2)
                if n_valid == 0 and V > 0:
                    n_valid = 1
                if n_invalid == 0 and I > 0:
                    n_invalid = 1

                assert n_valid + n_invalid == target_k, f"n_valid:{n_valid} + n_invalid:{n_invalid} != target_k:{target_k}"
                chosen = []
                chosen.extend(random.sample(valid_ids, k=n_valid))
                chosen.extend(random.sample(invalid_ids, k=n_invalid))
                random.shuffle(chosen)
                return chosen
            # >>> SPICE MOD END
            
# -------- while：TODO: translate comment
            remaining_indices = list(range(num_original))
            attempt = 0

# accepted_groups：TODO: translate comment
            # group = {
            #   "orig_idx": int,
            #   "qtype": str,
            #   "samples": List[Experience] (len=n),
# "valid_locals": List[int]（TODO: translate comment
            # }
            accepted_groups: List[Dict[str, Any]] = []

            while remaining_indices and attempt < max_retry:
                attempt += 1
                logger.info(f"[challenger] collect-valid-QA attempt {attempt}, remaining {len(remaining_indices)}/{num_original}")

                cur_qtypes = [question_types_full[i] for i in remaining_indices]
                cur_prompts = [all_final_prompts[i] for i in remaining_indices]
                cur_token_ids = [all_final_prompt_token_ids[i] for i in remaining_indices]

                expanded_prompts: List[str] = []
                expanded_token_ids: List[List[int]] = []
                expanded_qtypes: List[str] = []
                for qt, p, tid in zip(cur_qtypes, cur_prompts, cur_token_ids):
# >>> MOD: TODO: translate comment
                    for _ in range(challenger_n_candidates_per_prompt):
                        expanded_prompts.append(p)
                        expanded_token_ids.append(tid)
                        expanded_qtypes.append(qt)
                    # <<< MOD
                    # for _ in range(n_samples_per_prompt):
                    #     expanded_prompts.append(p)
                    #     expanded_token_ids.append(tid)
                    #     expanded_qtypes.append(qt)

                expanded_labels = [None] * len(expanded_prompts)

                all_outputs_this, samples_list_this = _run_vllm_and_build_samples(
                    expanded_prompts, expanded_labels, cur_prompt_token_ids=expanded_token_ids
                )


                sample_is_valid = [False] * len(samples_list_this)
                for k in range(len(samples_list_this)):
                    is_valid, _, _, _ = self._parse_and_validate_challenger_output(
                        question_type=expanded_qtypes[k],
                        output=all_outputs_this[k],
                        rollout_sample=samples_list_this[k],
                    )
                    sample_is_valid[k] = bool(is_valid)

                next_remaining = []
                for j, orig_idx in enumerate(remaining_indices):
                    # start = j * n_samples_per_prompt
                    # end = start + n_samples_per_prompt
                    # group_samples = samples_list_this[start:end]
                    # valid_locals = [local for local in range(n_samples_per_prompt) if sample_is_valid[start + local]]

                    # if not valid_locals:
                    #     next_remaining.append(orig_idx)
                    #     continue
# >>> MOD: TODO: translate comment
                    start = j * challenger_n_candidates_per_prompt
                    end = start + challenger_n_candidates_per_prompt

                    valid_locals_all = [local for local in range(challenger_n_candidates_per_prompt) if sample_is_valid[start + local]]
                    invalid_locals_all = [local for local in range(challenger_n_candidates_per_prompt) if not sample_is_valid[start + local]]

                    if not valid_locals_all:
                        next_remaining.append(orig_idx)
                        continue


                    keep_locals = _subsample_preserve_valid_invalid_ratio(valid_locals_all, invalid_locals_all, n_samples_per_prompt)
                    if not keep_locals:

                        next_remaining.append(orig_idx)
                        continue

                    group_samples = [samples_list_this[start + local] for local in keep_locals]

                    valid_locals = [idx_in_group for idx_in_group, local in enumerate(keep_locals) if sample_is_valid[start + local]]
                    # <<< MOD

                    accepted_groups.append({
                        "orig_idx": orig_idx,
                        "qtype": question_types_full[orig_idx],
                        "samples": group_samples,
                        "valid_locals": valid_locals,
                    })

                remaining_indices = next_remaining

            if remaining_indices:
                logger.info(f"[challenger] max_retry reached, dropping {len(remaining_indices)} prompts without any valid QA.")

            if not accepted_groups:
                if return_reasoner_experiences:
                    if return_generation_logs:
                        return [], [], [], []
                    return [], [], []
                if return_generation_logs:
                    return [], [], []
                return [], []

# -------- while TODO: translate comment

# max_qa_per_prompt_to_reasoner = int(kwargs.get("max_qa_per_prompt_to_reasoner", 0))  # 0 TODO: translate comment

            base_reasoner_prompts: List[str] = []
            base_reasoner_gold: List[str] = []
            base_group_idx: List[int] = []     # base b TODO: translate comment
            base_local_idx: List[int] = []     # base b TODO: translate comment
            base_prompt_for_norm: List[str] = []  


            group_local_to_base: List[Dict[int, int]] = []

# [NEW LOG] TODO: translate comment
            base_challenger_q: List[str] = []
            base_challenger_a: List[str] = []
            
            for g_idx, g in enumerate(accepted_groups):
                local_to_base: Dict[int, int] = {}
                valid_locals = g["valid_locals"]

                # if max_qa_per_prompt_to_reasoner > 0 and len(valid_locals) > max_qa_per_prompt_to_reasoner:
                #     valid_locals = random.sample(valid_locals, max_qa_per_prompt_to_reasoner)

                for local in valid_locals:
                    s = g["samples"][local]
                    q = s.questions[0]
                    a = s.answers[0]

                    rp = preprocess_data(
                        q,
                        apply_chat_template=self.apply_chat_template,
                        template_type=self.strategy.args.template_type,
                    )
                    b = len(base_reasoner_prompts)
                    base_reasoner_prompts.append(rp)
                    base_prompt_for_norm.append(rp)
                    base_reasoner_gold.append(a)
                    base_group_idx.append(g_idx)
                    base_local_idx.append(local)
                    local_to_base[local] = b
                    base_challenger_q.append(q)   # [NEW LOG]
                    base_challenger_a.append(a)   # [NEW LOG]
                    
                group_local_to_base.append(local_to_base)

            if not base_reasoner_prompts:
                if return_reasoner_experiences:
                    if return_generation_logs:
                        return [], [], [], []
                    return [], [], []
                if return_generation_logs:
                    return [], [], []
                return [], []

            base_token_ids = self.tokenize_fn(base_reasoner_prompts, self.prompt_max_len, padding=False)["input_ids"]

            expanded_reasoner_token_ids: List[List[int]] = []
            expanded_owner_b: List[int] = []
            for b, tid in enumerate(base_token_ids):
                for _ in range(num_reasoner_samples):
                    expanded_reasoner_token_ids.append(tid)
                    expanded_owner_b.append(b)

            sampling_params_reasoner = SamplingParams(
                temperature=kwargs.get("temperature", 1.0),
                top_p=kwargs.get("top_p", 1.0),
                top_k=kwargs.get("top_k", -1),
                max_tokens=reasoner_max_new_tokens,
                min_tokens=kwargs.get("min_new_tokens", 1),
                skip_special_tokens=kwargs.get("skip_special_tokens", False),
                include_stop_str_in_output=True,
                logprobs=1 if self.strategy.args.enable_vllm_is_correction else None,
            )

            reasoner_outputs = _run_vllm_outputs_only(expanded_reasoner_token_ids, sampling_params_reasoner)

            grouped_reasoner_outputs: Dict[int, List[Any]] = {}
            grouped_reasoner_texts: Dict[int, List[str]] = {}
            for out, b in zip(reasoner_outputs, expanded_owner_b):
                txt = out.outputs[0].text.strip() if out.outputs and out.outputs[0].text else ""
                grouped_reasoner_outputs.setdefault(b, []).append(out)
                grouped_reasoner_texts.setdefault(b, []).append(txt)

# -------- TODO: translate comment
            base_reward: Dict[int, float] = {}
            base_extreme: Dict[int, bool] = {}
# [NEW LOG] TODO: translate comment
            base_eval_detail: Dict[int, Dict[str, Any]] = {}
            for b in range(len(base_reasoner_prompts)):
                preds_raw = grouped_reasoner_texts.get(b, [])
                # if len(preds_raw) != num_reasoner_samples:
                #     preds_raw = (preds_raw + [""] * num_reasoner_samples)[:num_reasoner_samples]
                assert len(preds_raw) == num_reasoner_samples, f"Reasoner 推理次数 {len(preds_raw)} 一定等于 num_reasoner_samples {num_reasoner_samples}"

                rp = base_prompt_for_norm[b]
                gold = str(base_reasoner_gold[b]).strip()
                
                correct_flags: List[float] = []
                pred_ans = []
                for response in preds_raw:
                    ans, ok = self.math_reward_fn(response, gold)
                    pred_ans.append(ans)
                    correct_flags.append(1.0 if ok else 0.0)
                
                # pred_ans = [extract_all_boxed_expressions(t) for t in preds_raw]
# pred_ans = normalize_pred_answers(pred_ans, rp)  

                # correct_flags: List[float] = []
                # for ans in pred_ans:
                #     if not ans:
                #         correct_flags.append(0.0)
                #         continue
                #     ans_str = str(ans).strip()
                #     try:
                #         ok = verify(parse("$" + gold + "$"), parse("$" + ans_str + "$")) or gold.replace(" ", "") == ans_str.replace(" ", "")
                #     except Exception:
                #         ok = (gold.replace(" ", "") == ans_str.replace(" ", ""))
                #     correct_flags.append(1.0 if ok else 0.0)

                base_reward[b] = _compute_challenger_reward(correct_flags)
                base_extreme[b] = _is_extreme(correct_flags)
                base_eval_detail[b] = {  # [NEW LOG]
                    "gold": gold,
                    "reasoner_raw_texts": preds_raw,
                    "pred_answers_norm": pred_ans,
                    "verify_flags": correct_flags,
                }
                
# -------- TODO: translate comment
            final_challenger_samples: List[Experience] = []
            final_challenger_rewards: List[float] = []

            final_reasoner_samples: List[Experience] = []
            final_reasoner_outputs_selected: List[Any] = []
            final_reasoner_labels_selected: List[Any] = []
            


            path_to_max_reward: Dict[str, List[float]] = {}

            for g_idx, g in enumerate(accepted_groups):
                local_to_base = group_local_to_base[g_idx]
                if not local_to_base:
                    continue


                non_extreme_bs = [b for b in local_to_base.values() if not base_extreme.get(b, True)]
                if not non_extreme_bs:

                    continue


                best_b = max(non_extreme_bs, key=lambda b: base_reward.get(b, float(self.invalid_penalty)))

# (1) Challenger：TODO: translate comment
                group_rewards_this_path: List[float] = []
                for local in range(n_samples_per_prompt):
                    s = g["samples"][local]
                    if local in local_to_base:
                        b = local_to_base[local]
                        r = float(base_reward.get(b, float(self.invalid_penalty)))
                    else:
                        r = float(self.invalid_penalty)

                    final_challenger_samples.append(s)
                    final_challenger_rewards.append(r)
                    group_rewards_this_path.append(r)
                

                orig_idx = int(g["orig_idx"])
                path = str(original_paths[orig_idx])
                max_r = max(group_rewards_this_path) if group_rewards_this_path else float(self.invalid_penalty)
                if path not in path_to_max_reward:
                    path_to_max_reward[path] = [max_r]
                else:
                    path_to_max_reward[path].append(max_r)  #  = max(path_to_max_reward[path], max_r)

# (2) Reasoner：best_b TODO: translate comment
                outs_best = grouped_reasoner_outputs.get(best_b, [])
                assert len(outs_best) == num_reasoner_samples, f"Reasoner outputs for best_b {best_b} should be {num_reasoner_samples}, got {len(outs_best)}"
                # outs_best = (outs_best + [])[:num_reasoner_samples]
                rp_best = base_reasoner_prompts[best_b]
                gold_best = base_reasoner_gold[best_b]


                for out in outs_best:
                    exp_r = self._build_reasoner_experience_from_output(
                        out=out,
                        prompt_text=rp_best,
                        label=gold_best,
                        max_new_tokens=reasoner_max_new_tokens,
                    )
                    final_reasoner_samples.append(exp_r)
                    final_reasoner_outputs_selected.append(out)
                    final_reasoner_labels_selected.append(gold_best)
                
# [NEW LOG] TODO: translate comment
                if return_generation_logs:
                    detail = base_eval_detail.get(best_b, {})
                    generation_logs.append({
                        "orig_idx": int(g["orig_idx"]),
                        "qtype": str(g["qtype"]),
                        "best_b": int(best_b),
                        "challenger_question": base_challenger_q[best_b],
                        "challenger_answer": base_challenger_a[best_b],
                        "challenger_reward": float(base_reward.get(best_b, float(self.invalid_penalty))),
                        "reasoner_prompt": rp_best,  
                        "reasoner_gold": str(gold_best),
                        "reasoner_raw_texts": detail.get("reasoner_raw_texts", []),
                        "reasoner_pred_answers": detail.get("pred_answers_norm", []),
                        "verify_flags": detail.get("verify_flags", []),
                        "is_extreme": bool(base_extreme.get(best_b, True)),
                    })
                    

            if final_challenger_samples:
                rt = torch.tensor(final_challenger_rewards, dtype=torch.float32)
                rewards_info = [{
                    "rewards": rt,
                    "scores": rt,
                    "extra_logs": {"role": torch.full((len(final_challenger_samples),), 0.0)},
                }]
                update_samples_with_rewards(rewards_info, final_challenger_samples)


            if final_reasoner_samples:
                self.get_self_rewards(
                    final_reasoner_outputs_selected,
                    final_reasoner_labels_selected,
                    final_reasoner_samples,
                    role="reasoner",
                    **kwargs
                )

# [NEW LOG] TODO: translate comment
            if return_reasoner_experiences:
                if return_generation_logs:
                    return final_challenger_samples, final_reasoner_samples, path_to_max_reward, generation_logs
                return final_challenger_samples, final_reasoner_samples, path_to_max_reward
            else:
                if return_generation_logs:
                    return final_challenger_samples, path_to_max_reward, generation_logs
                return final_challenger_samples, path_to_max_reward

        # =========================
# reasoner TODO: translate comment
        # =========================
        if (not role) or role == "reasoner":

            chat_prompts = []
            for prompt in all_prompts:  
                chat_prompts.append(
                    preprocess_data(prompt, apply_chat_template=self.apply_chat_template, template_type=self.strategy.args.template_type)
                )
            all_prompts_expanded = sum([[prompt] * n_samples_per_prompt for prompt in chat_prompts], [])
            all_labels_expanded = sum([[label] * n_samples_per_prompt for label in all_labels], [])
        else:
            raise ValueError(f"Unknown role for vllm generation: {role}")

        all_outputs, samples_list = _run_vllm_and_build_samples(all_prompts_expanded, all_labels_expanded)

        self.get_self_rewards(all_outputs, all_labels_expanded, samples_list, role, **kwargs)
        return samples_list

# ====== TODO: translate comment
    def _build_reasoner_experience_from_output(self, out, prompt_text: str, label: Any, max_new_tokens: int) -> "Experience":
        input_ids = list(out.prompt_token_ids) + list(out.outputs[0].token_ids)
        attention_mask = [1] * len(input_ids)

        sequences = torch.tensor(input_ids, dtype=torch.long)
        attention_mask_t = torch.tensor(attention_mask, dtype=torch.long)

        action_mask = torch.zeros_like(attention_mask_t)
        resp_len = len(out.outputs[0].token_ids)
        action_mask[len(out.prompt_token_ids) : len(out.prompt_token_ids) + resp_len] = 1

        rollout_log_probs = None
        if self.strategy.args.enable_vllm_is_correction:
            lp = []
            response_ids = list(out.outputs[0].token_ids)
            for j, logprob in enumerate(out.outputs[0].logprobs):
                lp.append(logprob[response_ids[j]].logprob)
            rollout_log_probs = torch.tensor([0.0] * len(list(out.prompt_token_ids)) + lp)
            rollout_log_probs = rollout_log_probs[1 : self.prompt_max_len + max_new_tokens].to("cpu")

        trunc_len = self.prompt_max_len + max_new_tokens
        sequences = sequences[:trunc_len].to("cpu")
        attention_mask_t = attention_mask_t[:trunc_len].to("cpu")
        action_mask = action_mask[1:trunc_len].to("cpu")

        info = {
            "response_length": torch.tensor([resp_len]),
            "total_length": torch.tensor([attention_mask_t.float().sum()]),
            "response_clip_ratio": torch.tensor([resp_len >= max_new_tokens]),
        }

        return Experience(
            sequences=sequences.unsqueeze(0),
            attention_mask=attention_mask_t.unsqueeze(0),
            action_mask=action_mask.unsqueeze(0),
            rollout_log_probs=rollout_log_probs.unsqueeze(0) if rollout_log_probs is not None else None,
            prompts=[prompt_text],
            labels=[label],
            info=info,
        )
        
    
    def _generate_vllm_cold_start(self, all_prompts: List[str], all_labels, role: str, paths: List[str], **kwargs) -> List[Experience]:
        """Generate samples using vLLM engine.

        Args:
            all_prompts: List of prompts to generate from
            all_labels: List of labels corresponding to prompts
            role: Role for experience making (e.g., "challenger", "reasoner")
            **kwargs: Additional arguments for generation

        Returns:
            (samples_list, paths_list): paths_list 与 samples_list 一一对应
        """
        llms = self.vllm_engines
        args = self.strategy.args
        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
            logprobs=1 if self.strategy.args.enable_vllm_is_correction else None,
        )
        max_response_length = kwargs.get("max_new_tokens", 1024)
        truncate_length = self.prompt_max_len + max_response_length

        # Expand prompt list based on the number of samples per prompt
        n_samples_per_prompt = kwargs.pop("n_samples_per_prompt", args.n_samples_per_prompt)
        

        def _run_vllm_and_build_samples(cur_prompts: List[str], cur_labels: List[Any], cur_prompt_token_ids: Optional[List[List[int]]] = None,):
            if not cur_prompts:  
                return [], []
# ===== TODO: translate comment
            if cur_prompt_token_ids is None:
                all_prompt_token_ids = self.tokenize_fn(cur_prompts, self.prompt_max_len, padding=False)["input_ids"]
            else:
                all_prompt_token_ids = cur_prompt_token_ids

                assert len(all_prompt_token_ids) == len(cur_prompts)


            refs = []
            batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms)
            for i, llm in enumerate(llms):
                prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]
                if not prompt_token_ids:
                    continue
                refs.append(llm.add_requests.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids))
            ray.get(refs)

            # Retrieve and combine results from all outputs
            all_output_refs = []
            for i, llm in enumerate(llms):
                all_output_refs.append(llm.get_responses.remote())
            all_outputs = sum(ray.get(all_output_refs), [])


            samples_list: List[Experience] = []
            for i in range(len(all_outputs)):
                output = all_outputs[i]
                prompt = cur_prompts[i]
                label = cur_labels[i]


                input_ids = list(output.prompt_token_ids) + list(output.outputs[0].token_ids)
                attention_mask = [1] * len(input_ids)

                sequences = torch.tensor(input_ids, dtype=torch.long)
                attention_mask = torch.tensor(attention_mask)

                # Create action mask based on output token positions
                action_mask = torch.zeros_like(attention_mask)
                response_length = len(output.outputs[0].token_ids)
                action_mask[len(output.prompt_token_ids) : len(output.prompt_token_ids) + response_length] = 1

# rollout_log_probs（TODO: translate comment
                rollout_log_probs = None
                if self.strategy.args.enable_vllm_is_correction:
                    rollout_log_probs = []
                    response_ids = list(output.outputs[0].token_ids)
                    for j, logprob in enumerate(output.outputs[0].logprobs):
                        rollout_log_probs.append(logprob[response_ids[j]].logprob)

                    rollout_log_probs = torch.tensor([0.0] * len(list(output.prompt_token_ids)) + rollout_log_probs)
                    rollout_log_probs = rollout_log_probs[1:truncate_length].to("cpu")

                sequences = sequences[:truncate_length].to("cpu")
                attention_mask = attention_mask[:truncate_length].to("cpu")
                action_mask = action_mask[1:truncate_length].to("cpu")
                total_length = attention_mask.float().sum()
                is_clipped = response_length >= max_response_length

                info = {
                    "response_length": torch.tensor([response_length]),
                    "total_length": torch.tensor([total_length]),
                    "response_clip_ratio": torch.tensor([is_clipped]),
                }

                rollout_samples = Experience(
                    sequences=sequences.unsqueeze(0),
                    attention_mask=attention_mask.unsqueeze(0),
                    action_mask=action_mask.unsqueeze(0),
                    rollout_log_probs=rollout_log_probs.unsqueeze(0) if rollout_log_probs is not None else None,
                    prompts=[prompt],
                    labels=[label],
                    info=info,
                )
                samples_list.append(rollout_samples)
            return all_outputs, samples_list
        
        # =========================
# challenger TODO: translate comment
        # =========================
        if role == "challenger":

            max_retry = kwargs.get("challenger_max_resample_attempts", getattr(args, "challenger_max_resample_attempts", 3))
            original_prompts = all_prompts
            original_paths = paths
            num_original = len(original_prompts)
# challenger TODO: translate comment
            original_labels = [None] * num_original
            
            # >>> SPICE MOD START: oversample then subsample-to-G

            challenger_n_candidates_per_prompt = kwargs.get(
                "challenger_n_candidates_per_prompt",
                getattr(args, "challenger_n_candidates_per_prompt", None),
            )
            if challenger_n_candidates_per_prompt is None:
                challenger_n_candidates_per_prompt = int(round(n_samples_per_prompt * 4))
            challenger_n_candidates_per_prompt = max(int(challenger_n_candidates_per_prompt), int(n_samples_per_prompt) * 2)
            # >>> SPICE MOD END
            
            # >>> SPICE MOD START: helper for ratio-preserving subsample
            def _subsample_preserve_valid_invalid_ratio(valid_ids: List[int], invalid_ids: List[int], target_k: int) -> List[int]:
                """
                Return a list of global indices of length target_k (when possible),
                preserving valid:invalid ratio, and if both exist, keep at least 1 from each.
                """
                V, I = len(valid_ids), len(invalid_ids)
                M = V + I
                if target_k <= 0 or M == 0:
                    return []
                if target_k >= M:
                    # Already have <= target_k candidates
                    return list(valid_ids) + list(invalid_ids)

                # Prefer valid if only one class exists
                if I == 0:
                    return random.sample(valid_ids, k=min(target_k, V))
                if V == 0:
                    # caller should handle resample; keep empty to indicate failure
                    return []
                if target_k == 1:
                    # Can't keep both classes; prefer valid
                    return [random.choice(valid_ids)]
                # Proportional target counts
                n_valid = int(round(target_k * V / M))
                n_valid = max(1, min(target_k - 1, n_valid))
                n_invalid = target_k - n_valid

                # Availability adjustment
                n_valid = min(n_valid, V)
                n_invalid = min(n_invalid, I)

                # Ensure at least 1 from each when both exist (and target_k>=2)
                if n_valid == 0 and V > 0:
                    n_valid = 1
                if n_invalid == 0 and I > 0:
                    n_invalid = 1
                assert n_valid + n_invalid == target_k, f"n_valid:{n_valid} + n_invalid:{n_invalid} != target_k:{target_k}"
                chosen = []
                chosen.extend(random.sample(valid_ids, k=n_valid))
                chosen.extend(random.sample(invalid_ids, k=n_invalid))
                random.shuffle(chosen)
                return chosen
            # >>> SPICE MOD END
            
            if self.judge_question_type:

                question_types_full = self.get_question_type(original_prompts, sampling_params)  # TODO: TODO: translate comment
            else:
                question_types_full = ['mcq'] * num_original  
            assert len(question_types_full) == num_original, "question_types length mismatch"
            
            all_final_prompts: List[str] = []
            for ques_type, content, path in zip(question_types_full, original_prompts, original_paths):
                if self.generate_qa_prompt == "math":
                    if "mcq" in ques_type.lower():
                        p = MCQ_PROMPT_WITH_PATH_MATH.replace("{text}", content).replace("{path}", path)
                    else:
                        p = FREE_PROMPT_WITH_PATH_MATH.replace("{text}", content).replace("{path}", path)
                elif self.generate_qa_prompt == "med":
                    if "mcq" in ques_type.lower():
                        p = MCQ_PROMPT_WITH_PATH_MEDICINE.replace("{text}", content).replace("{path}", path)
                    else:
                        p = FREE_PROMPT_WITH_PATH_MEDICINE.replace("{text}", content).replace("{path}", path)
                elif self.generate_qa_prompt == "phy":
                    if "mcq" in ques_type.lower():
                        p = MCQ_PROMPT_WITH_PATH_PHYSICS.replace("{text}", content).replace("{path}", path)
                    else:
                        p = FREE_PROMPT_WITH_PATH_PHYSICS.replace("{text}", content).replace("{path}", path)
                else:
                    raise ValueError(f"Unknown generate_qa_prompt: {self.generate_qa_prompt}")
                p = preprocess_challenger_data(p, apply_chat_template=self.apply_chat_template, template_type=self.strategy.args.template_type)
                all_final_prompts.append(p)

            all_final_prompt_token_ids: List[List[int]] = self.tokenize_fn(
                all_final_prompts, self.prompt_max_len, padding=False
            )["input_ids"]
            
            truncated_flags = [len(token_ids) >= self.prompt_max_len for token_ids in all_final_prompt_token_ids]
            num_truncated = sum(truncated_flags)
            logger.info(f"{num_truncated}/{len(truncated_flags)} prompts truncated (len >= max_length)")
            assert num_truncated == 0, "Some prompts are too long and get truncated! Should increase self.instruction_len"
            
            remaining_indices = list(range(num_original))
            
            accepted_samples_per_orig: List[Optional[List[Experience]]] = [None] * num_original
            accepted_outputs_per_orig: List[Optional[List[Any]]] = [None] * num_original

            attempt = 0
            while remaining_indices and attempt < max_retry:
                attempt += 1
                logger.info(
                    f"[challenger] sampling attempt {attempt}, "
                    f"remaining {len(remaining_indices)} / {num_original} prompts"
                )
                
                cur_question_types = [question_types_full[i] for i in remaining_indices]
                cur_final_prompts = [all_final_prompts[i] for i in remaining_indices]
                cur_final_token_ids = [all_final_prompt_token_ids[i] for i in remaining_indices]
                
                expanded_prompts: List[str] = []
                expanded_token_ids: List[List[int]] = []
                for p, token_ids in zip(cur_final_prompts, cur_final_token_ids):
                    for _ in range(challenger_n_candidates_per_prompt):
                        expanded_prompts.append(p)
                        expanded_token_ids.append(token_ids)
                expanded_labels = [None] * len(expanded_prompts)
                self.challenger_question_types = sum(
                    [[ques_type] * challenger_n_candidates_per_prompt for ques_type in cur_question_types],
                    []
                )
                
                all_outputs_this, samples_list_this = _run_vllm_and_build_samples(
                    expanded_prompts, 
                    expanded_labels,
                    cur_prompt_token_ids=expanded_token_ids,
                )
                next_remaining_indices = []

                for j, orig_idx in enumerate(remaining_indices):
                    start = j * challenger_n_candidates_per_prompt
                    end = start + challenger_n_candidates_per_prompt

                    valid_ids: List[int] = []
                    invalid_ids: List[int] = []

                    for k in range(start, end):
                        qtype = self.challenger_question_types[k]
                        is_valid, _, _, _ = self._parse_and_validate_challenger_output(
                            question_type=qtype,
                            output=all_outputs_this[k],
                            rollout_sample=samples_list_this[k],
                        )
                        if is_valid:
                            valid_ids.append(k)
                        else:
                            invalid_ids.append(k)

                    if len(valid_ids) == 0:
                        next_remaining_indices.append(orig_idx)
                        continue

                    keep_ids = _subsample_preserve_valid_invalid_ratio(valid_ids, invalid_ids, n_samples_per_prompt)
                    if not keep_ids:
                        next_remaining_indices.append(orig_idx)
                        continue

                    accepted_samples_per_orig[orig_idx] = [samples_list_this[k] for k in keep_ids]
                    accepted_outputs_per_orig[orig_idx] = [all_outputs_this[k] for k in keep_ids]

                logger.info(
                    f"[challenger] attempt {attempt} done, "
                    f"newly accepted groups: {len(remaining_indices) - len(next_remaining_indices)}, "
                    f"still remaining: {len(next_remaining_indices)}"
                )
                remaining_indices = next_remaining_indices
                    

            if remaining_indices:
                logger.info(
                    f"[challenger] max_retry={max_retry} reached, "
                    f"dropping {len(remaining_indices)} prompts without any valid output."
                )

            
            final_samples_list: List[Experience] = []
            final_outputs: List[Any] = []
            final_question_types: List[str] = []
            final_paths = []  

            for orig_idx in range(num_original):
                group_samples = accepted_samples_per_orig[orig_idx]
                if group_samples is None:
                    continue  
                group_outputs = accepted_outputs_per_orig[orig_idx]
                assert group_outputs is not None
                final_samples_list.extend(group_samples)
                final_outputs.extend(group_outputs)

                final_question_types.extend(
                    [question_types_full[orig_idx]] * len(group_samples)
                )
                final_paths.extend(
                    [original_paths[orig_idx]] * len(group_samples)
                )


            if not final_samples_list:
                return [], 
            self.challenger_question_types = final_question_types
            final_labels = [None] * len(final_samples_list)
            self.get_self_rewards(final_outputs, final_labels, final_samples_list, role, **kwargs, )
            return final_samples_list, final_paths

        if (not role) or role == "reasoner":

            chat_prompts = []
            for prompt in all_prompts:  
                chat_prompts.append(
                    preprocess_data(prompt, apply_chat_template=self.apply_chat_template, template_type=self.strategy.args.template_type)
                )
            all_prompts_expanded = sum([[prompt] * n_samples_per_prompt for prompt in chat_prompts], [])
            all_labels_expanded = sum([[label] * n_samples_per_prompt for label in all_labels], [])
        else:
            raise ValueError(f"Unknown role for vllm generation: {role}")

        all_outputs, samples_list = _run_vllm_and_build_samples(all_prompts_expanded, all_labels_expanded)

        self.get_self_rewards(all_outputs, all_labels_expanded, samples_list, role, **kwargs)
        return samples_list, []

    def _validate_language(
        self,
        content: str,
        print_str: str = "",
        max_non_english_word_ratio: float = 0.2,  
        min_non_english_words: int = 1,
    ) -> bool:

        _CJK_RE = re.compile(
            r"[\u4E00-\u9FFF"      # CJK Unified Ideographs
            r"\u3400-\u4DBF"       # CJK Unified Ideographs Extension A
            r"\uF900-\uFAFF"       # CJK Compatibility Ideographs
            r"\U00020000-\U0002EBEF"  # Extension B..F (covers a large range)
            r"]"
        )
        if not content:
            print(f"[Verification Failed] {print_str} is empty")
            return False

        s = unicodedata.normalize("NFKC", content)
        if _CJK_RE.search(s):
            print(f"[Verification Failed] {print_str} contains Chinese/Hanzi: {content}")
            return False
        def is_ascii_english_letter(ch: str) -> bool:
            o = ord(ch)
            return (65 <= o <= 90) or (97 <= o <= 122)  # A-Z a-z
        def extract_words(text: str) -> List[str]:
            """Word: continuous segment composed of letters; allows - and ' as internal connectors."""
            words = []
            buf = []
            n = len(text)
            i = 0
            while i < n:
                ch = text[i]
                if ch.isalpha():
                    buf.append(ch)
                    i += 1
                    continue
                if ch in ("'", "-") and buf and (i + 1) < n and text[i + 1].isalpha():
                    buf.append(ch)
                    i += 1
                    continue
                if buf:
                    words.append("".join(buf))
                    buf = []
                i += 1
            if buf:
                words.append("".join(buf))
            return words
        def is_math_letter(ch: str) -> bool:
            o = ord(ch)
            return (
                (0x0370 <= o <= 0x03FF) or   # Greek and Coptic (λ, μ, π, ...)
                (0x1F00 <= o <= 0x1FFF) or   # Greek Extended
                (0x2100 <= o <= 0x214F) or   # Letterlike Symbols (ℝ, ℤ, ℕ, ...)
                (0x1D400 <= o <= 0x1D7FF)    # Mathematical Alphanumeric Symbols (𝑥, 𝐀, 𝒩, ...)
            )
        def is_math_token(w: str) -> bool:
            """
            General Mathematics Tokens:
                - The token must contain at least 1 "mathematical letter" (Greek/Letterlike/Math Alnum).
                - And the token's total length is not too long (variables/symbols are generally short).
            This covers: δ, ω, λ, ω0, ΔG, μm, ℝ2, 𝑥1, etc.
            """
            if len(w) > 6:   
                return False
            has_math_letter = any(c.isalpha() and is_math_letter(c) for c in w)
            if not has_math_letter:
                return False
            for c in w:
                if c.isalpha():
                    continue
                if c.isdigit():
                    continue
                if c in ("'", "-"):
                    continue
                return False
            return True
        def is_english_word(w: str) -> bool:
            letters = [c for c in w if c.isalpha()]
            return bool(letters) and all(is_ascii_english_letter(c) for c in letters)
        words = extract_words(s)
        if not words:
            return True

        filtered_words = [w for w in words if not is_math_token(w)]
        if not filtered_words:
            return True
        non_eng = [w for w in filtered_words if not is_english_word(w)]
        ratio = len(non_eng) / len(filtered_words)
        if len(non_eng) >= min_non_english_words and ratio > max_non_english_word_ratio:
            examples = ", ".join(non_eng[:8])
            print(
                f"[Verification Failed] {print_str}: Proportion of Non-English Words={ratio:.3f} "
                f"({len(non_eng)}/{len(filtered_words)})，Example non-English word: {examples} | Original text: {content}"
            )
            return False
        return True
    
    def _parse_and_validate_challenger_output(
        self,
        question_type: str,
        output,
        rollout_sample,  # Experience
    ) -> Tuple[bool, Optional[str], Optional[str], str]:
        """
        Analyze the output of a single challenger sample and perform all validity checks.
        Return:
            is_valid: Whether all checks have passed
            question: If valid, the extracted question (otherwise None)
            answer: If valid, the extracted answer (otherwise None)
            test_output: The test_output returned by parse_json_output, for logging purposes
        """
        QUESTIONS_FORBIDDEN = ["answer", "answer:", "hardening_process", "multi-step", "conversions", "section", "section.", "section,",
                               "hints", "hints:", "hints.", "quotes", "document", "document?", "document.", "document," , "figure"]
        ANSWERS_FORBIDDEN = ["answer", "answer:", "answer,", "answer.",
                                "integer", "integer:", "integer,", "integer.",
                                "number", "number:", "number,", "number.",
                                "expression", "expression:", "expression,", "expression.",
                                "string", "string:", "string,", "string.",
                                "theorem", "theorem:", "theorem,", "theorem.",
                            ]
        QUESTIONS_FORBIDDEN_STR = ["the text", "in the passage", "based on the given", ]
        raw_text = output.outputs[0].text.strip()
        output_dict, test_output = parse_json_output(raw_text)
        if not output_dict:
            return False, None, None, test_output

        generated_question = output_dict.get("question") or output_dict.get("exam_question") or ""
        generated_answer = output_dict.get("correct_answer") or output_dict.get("answer") or ""
        if not isinstance(generated_question, str) or not isinstance(generated_answer, str):
            return False, None, None, test_output
        
        generated_question = generated_question.strip()
        generated_answer = generated_answer.strip()
        if not generated_question or not generated_answer or len(generated_question) < 50:
            return False, None, None, test_output

        q = generated_question.lower().split()
        a = generated_answer.lower().split()
        if any(w in q for w in QUESTIONS_FORBIDDEN) \
            or any(w in a for w in ANSWERS_FORBIDDEN):
            return False, None, None, test_output
        
        # based on the given parameters in the passage?
        if any(w in generated_question.lower() for w in QUESTIONS_FORBIDDEN_STR):
            return False, None, None, test_output

        if "mcq" in question_type.lower():
            if (not is_mcq_question_text(generated_question)
                    or not is_valid_mcq_answer_letter(generated_answer)):
                return False, None, None, test_output

        if not isinstance(generated_question, str) or not isinstance(generated_answer, str):
            return False, None, None, test_output

        if self._validate_language(generated_question, print_str=f"生成的问题") is False:
            return False, None, None, test_output

        if "free" in question_type.lower() and (
            not is_valid_free_qa_from_model_output(
                generated_question,
                generated_answer,
                min_question_len=50,
                max_answer_tokens=5,
            )
        ):
            return False, None, None, test_output


        rollout_sample.questions = [generated_question]
        rollout_sample.answers = [generated_answer]
        return True, generated_question.strip(), generated_answer.strip(), test_output
    
    
    def get_self_rewards(self, all_outputs, all_labels, samples_list, role, **kwargs):
        llms = self.vllm_engines
        num_reasoner_samples = kwargs.get("num_reasoner_samples", 4)  
        num_samples = len(all_outputs)
        rewards: List[float] = [0.0] * num_samples

        if role == "challenger":
            assert len(self.challenger_question_types) == num_samples, "challenger_question_types length mismatch"
            reasoner_prompts: List[str] = []
            reasoner_indices: List[int] = []
            standard_answers: Dict[int, str] = {}  # idx -> TODO: translate comment
            
            challenger_model_outputs = ""
            for i in range(num_samples):
                question_type = self.challenger_question_types[i]
                output = all_outputs[i]
                rollout_sample = samples_list[i]  
                questions = rollout_sample.questions
                answers = rollout_sample.answers

                if questions is None or answers is None or len(questions) == 0 or len(answers) == 0:
                    rewards[i] = float(self.invalid_penalty)
                    continue
                generated_question = questions[0]
                generated_answer = answers[0]

                if not isinstance(generated_question, str) or not isinstance(generated_answer, str) \
                        or not generated_question.strip() or not generated_answer.strip():
                    rewards[i] = float(self.invalid_penalty)
                    continue

                challenger_model_outputs += (
                    "="*10 + f"Challenger Sample Prompt {i}th Start" + "="*10 + "\n"
                    + str(rollout_sample.prompts) + "\n"
                    + "="*10 + f"Challenger Sample Prompt {i}th End" + "="*10 + "\n"
                    + f"\n[reuse validated QA] Q: {generated_question}\nA: {generated_answer}\n"
                )

                prompt = preprocess_data(generated_question, apply_chat_template=self.apply_chat_template, template_type=self.strategy.args.template_type)
                reasoner_prompts.append(prompt)
                reasoner_indices.append(i)
                standard_answers[i] = generated_answer.strip()

            if self.challenger_output_log_path:
                with open(self.challenger_output_log_path, "a", encoding="utf-8") as f:
                    f.write(f"{challenger_model_outputs}\n\n")
            

            if not reasoner_prompts:
                print("[get_self_rewards][challenger] All challenger samples invalid, applying penalties.")
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
                rewards_info = [{
                    "rewards": rewards_tensor,
                    "scores": rewards_tensor,
                    "extra_logs": {
                        "role": torch.tensor([0]*num_samples)
                    },
                }]
                update_samples_with_rewards(rewards_info, samples_list)
                return rewards_tensor
            
            print(f"[get_self_rewards][challenger] Generating reasoner samples for {len(reasoner_prompts)} prompts...")
            expanded_prompts = []
            expanded_owner_indices = []  
            
            reasoner_idx_map_to_prompts = {}
            for idx, p in zip(reasoner_indices, reasoner_prompts):
                reasoner_idx_map_to_prompts[idx] = p
                for _ in range(num_reasoner_samples):
                    expanded_prompts.append(p)
                    expanded_owner_indices.append(idx)

            sampling_params = SamplingParams(
                temperature=kwargs.get("temperature", 1.0),
                top_p=kwargs.get("top_p", 1.0),
                top_k=kwargs.get("top_k", -1),
                max_tokens=kwargs.get("max_new_tokens", 512),
                min_tokens=kwargs.get("min_new_tokens", 1),
                skip_special_tokens=kwargs.get("skip_special_tokens", False),
                include_stop_str_in_output=True,
                logprobs=1 if self.strategy.args.enable_vllm_is_correction else None,
            )
            expanded_token_ids = self.tokenize_fn(
                expanded_prompts, self.prompt_max_len, padding=False
            )["input_ids"]
            
            refs = []
            batch_size = (len(expanded_token_ids) + len(llms) - 1) // len(llms)
            for eng_id, llm in enumerate(llms):
                sub_token_ids = expanded_token_ids[
                    eng_id * batch_size : (eng_id + 1) * batch_size
                ]
                if not sub_token_ids:
                    continue
                refs.append(
                    llm.add_requests.remote(
                        sampling_params=sampling_params,
                        prompt_token_ids=sub_token_ids,
                    )
                )
            ray.get(refs)

            output_refs = [llm.get_responses.remote() for llm in llms]
            reasoner_outputs = sum(ray.get(output_refs), [])
            
            if len(reasoner_outputs) != len(expanded_prompts):
                print(
                    f"[get_self_rewards][challenger] WARNING: outputs={len(reasoner_outputs)}, prompts={len(expanded_prompts)}"
                )

            grouped_results = {}
            for out, owner_idx in zip(reasoner_outputs, expanded_owner_indices):
                text = out.outputs[0].text.strip()
                if text:
                    grouped_results.setdefault(owner_idx, []).append(text)
                else:
                    grouped_results.setdefault(owner_idx, []).append("没有输出")
                    
            for idx in reasoner_indices:
                gold_ans = standard_answers[idx]
                preds = grouped_results.get(idx, [])

                if not preds:
                    print(f"[get_self_rewards][challenger] Sample {idx} has no valid reasoner outputs.")
                    rewards[idx] = 0.0
                    continue

                correct_cnt: List[float] = []
                pred_ans = []
                for response in preds:
                    ans, ok = self.math_reward_fn(response, gold_ans)
                    pred_ans.append(ans)
                    correct_cnt.append(1.0 if ok else 0.0)

                assert len(correct_cnt) == num_reasoner_samples, f"Sample {idx} should have {num_reasoner_samples} predictions, got {len(preds)}"

                if self.args.challenger_rm_mode.lower() == "spice":
                    # ============= SPICE =============
                    var = np.var(np.asarray(correct_cnt, dtype=float))
                    reward = np.exp(- (var - 0.25) ** 2 / (2 * 0.01))
                elif self.args.challenger_rm_mode.lower() == "r-zero":
                    # ============ R-Zero =============
                    p_hat = sum(correct_cnt) / len(preds)
                    reward = 1.0 - 2.0 * abs(p_hat - 0.5)
                else:
                    raise ValueError(f"Unknown challenger_rm_mode: {self.args.challenger_rm_mode}")
                
                rewards[idx] = float(reward)

            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
            rewards_info = [{
                "rewards": rewards_tensor,
                "scores": rewards_tensor,
                "extra_logs": {
                    "role": torch.full((num_samples,), 0.0),
                },
            }]
            update_samples_with_rewards(rewards_info, samples_list)
            return rewards_tensor
        
        elif role == "reasoner":
            for i in range(num_samples):
                raw_text = all_outputs[i].outputs[0].text.strip()
                gold = str(all_labels[i]).strip()
                # pred = extract_all_boxed_expressions(raw_text)
                pred, ok = self.math_reward_fn(raw_text, gold)
                if not ok:
                    rewards[i] = 0.0
                else:
                    rewards[i] = 1.0

            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
            rewards_info = [{
                "rewards": rewards_tensor,
                "scores": rewards_tensor,
                "extra_logs": {
                    "role": torch.full((num_samples,), 1.0),
                },
            }]
            update_samples_with_rewards(rewards_info, samples_list)
            return rewards_tensor
        else:
            raise ValueError(f"Unknown role for get_self_rewards: {role}")


class RemoteExperienceMaker(ABC):
    def __init__(
        self,
        actor_model_group: RayActorGroup,
        critic_model_group: RayActorGroup,
        reward_model_group: RayActorGroup,
        initial_model_group: RayActorGroup,
        kl_controller,
        strategy=None,
        tokenizer=None,
        remote_reward_model=None,
        **kwargs,
    ):
        super().__init__()

        self.actor_model_group = actor_model_group
        self.critic_model_group = critic_model_group
        self.reward_model_group = reward_model_group
        self.initial_model_group = initial_model_group
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.advantage_estimator = strategy.args.advantage_estimator
        self.args = strategy.args

        # remote_rm_url indicates that the remote reward model is agent environment, remote http server or custom reward func
        self.remote_rm_url = self.args.remote_rm_url
        self.remote_reward_model = remote_reward_model
        self.tokenizer = tokenizer

    def split_rollout_samples(self, rollout_samples):
        for i, sample in enumerate(rollout_samples):
            sample.index = [i]

        samples_list = []
        if self.args.use_dynamic_batch:
            total_lengths = [int(s.info["total_length"].item()) for s in rollout_samples]
            effective_actor_num = (
                self.args.actor_num_nodes
                * self.args.actor_num_gpus_per_node
                // self.args.ring_attn_size
                // self.args.ds_tensor_parallel_size
            )
            minimum_batch_num = get_minimum_num_micro_batch_size(
                total_lengths,
                self.args.rollout_max_tokens_per_gpu,
                self.args.ring_attn_size,
                self.args.ds_tensor_parallel_size,
            )
            minimum_batch_num = minimum_batch_num // effective_actor_num * effective_actor_num
            num_batch = max(minimum_batch_num, effective_actor_num)
            batch_indexes = get_seqlen_balanced_partitions(total_lengths, num_batch, False)
            for micro_index in batch_indexes:
                micro_batch = [rollout_samples[idx] for idx in micro_index]
                concat_samples = Experience.concat_experiences(micro_batch, self.tokenizer.pad_token_id)
                samples_list.append(concat_samples)
        else:
            batch_size = self.args.micro_rollout_batch_size
            for i in range(0, len(rollout_samples), batch_size):
                concat_samples = Experience.concat_experiences(
                    rollout_samples[i : i + batch_size], self.tokenizer.pad_token_id
                )
                samples_list.append(concat_samples)
        return samples_list

    @torch.no_grad()
    def make_experience_batch(self, rollout_samples) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        # Each batch of samples will be scheduled to a effective Ray Actor (i.e, a DP rank)
        samples_list = self.split_rollout_samples(rollout_samples)

        # Make experiences (models forward: logprobs, values, rewards, and kl divergence)
        experiences = self.make_experience(samples_list)

        # Process experiences (reward shaping, etc.)
        experiences = self.compute_advantages_and_returns(experiences)
        return experiences

    @torch.no_grad()
    def make_experience(self, samples_list: List[Experience]) -> List[Experience]:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        start_time = time.time()
        logger.info(f"🚀 Starting experience making with {sum([len(s.sequences) for s in samples_list])} samples")

        args = self.strategy.args
        device = "cpu"

        # Extract all information from samples in one pass
        # Convert samples into lists of tensors and metadata for batch processing
        sequences_list = [s.sequences for s in samples_list]
        attention_mask_list = [s.attention_mask for s in samples_list]
        action_mask_list = [s.action_mask for s in samples_list]

        # The rewards are already filled in the samples_list, such as the agent's environment rewards
        if samples_list[0].rewards is not None:
            r_refs = None
            pass
        elif self.remote_rm_url:
            queries_list = sum(
                [
                    self.tokenizer.batch_decode(remove_pad_token(seq, attention_mask), skip_special_tokens=False)
                    for seq, attention_mask in zip(sequences_list, attention_mask_list)
                ],
                [],
            )
            prompts_list = sum([s.prompts for s in samples_list], [])
            labels_list = sum([s.labels for s in samples_list], [])
            # Keep the remote call asynchronous
            r_refs = self.remote_reward_model.get_rewards.remote(queries_list, prompts_list, labels_list)
        else:
            # Batch call reward model
            r_refs = self.reward_model_group.async_run_method_batch(
                method_name="forward",
                sequences=sequences_list,
                attention_mask=attention_mask_list,
                pad_sequence=[True] * len(samples_list),
            )
        # Sync to avoid GPU OOM when colocate models
        if r_refs is not None and args.colocate_all_models and not self.remote_rm_url:
            ray.get(r_refs)
            ray.get(self.reward_model_group.async_run_method(method_name="empty_cache"))
                
        # Batch call actor model
        action_log_probs_ref = self.actor_model_group.async_run_method_batch(
            method_name="forward",
            sequences=sequences_list,
            action_mask=action_mask_list,
            attention_mask=attention_mask_list,
        )

        # Sync to avoid GPU OOM when colocate models
        if args.colocate_all_models or args.colocate_actor_ref:
            ray.get(action_log_probs_ref)
            ray.get(self.actor_model_group.async_run_method(method_name="empty_cache"))

        # Batch call critic model
        if self.critic_model_group is not None:
            if args.colocate_critic_reward and not self.remote_rm_url:
                ray.get(r_refs)
                ray.get(self.reward_model_group.async_run_method(method_name="empty_cache"))

            value_ref = self.critic_model_group.async_run_method_batch(
                method_name="forward",
                sequences=sequences_list,
                action_mask=action_mask_list,
                attention_mask=attention_mask_list,
            )
            if args.colocate_all_models or args.colocate_critic_reward:
                ray.get(value_ref)
                ray.get(self.critic_model_group.async_run_method(method_name="empty_cache"))
        else:
            value_ref = ray.put([[None]] * (len(samples_list) * args.ring_attn_size * args.ds_tensor_parallel_size))

        # Batch call initial model
        if self.initial_model_group is not None:
            base_action_log_probs_ref = self.initial_model_group.async_run_method_batch(
                method_name="forward",
                sequences=sequences_list,
                action_mask=action_mask_list,
                attention_mask=attention_mask_list,
            )

            if args.colocate_all_models or args.colocate_actor_ref:
                ray.get(base_action_log_probs_ref)
                ray.get(self.initial_model_group.async_run_method(method_name="empty_cache"))
        else:
            base_action_log_probs_ref = ray.put(
                [[None]] * (len(samples_list) * args.ring_attn_size * args.ds_tensor_parallel_size)
            )

        # Wait for all remote calls to complete and flatten the results
        # Note: the results duplicated ring_attn_size * ds_tensor_parallel_size times
        # This is because the actors in ring group and tp group will return the same output
        duplicate_factor = args.ring_attn_size * args.ds_tensor_parallel_size
        action_log_probs_list = sum(ray.get(action_log_probs_ref)[::duplicate_factor], [])
        base_action_log_probs_list = sum(ray.get(base_action_log_probs_ref)[::duplicate_factor], [])
        value_list = sum(ray.get(value_ref)[::duplicate_factor], [])

        # Process rewards based on source
        if samples_list[0].rewards is not None:
            pass
        elif self.remote_rm_url:
            # Get rewards info from remote model
            rewards_info = ray.get(r_refs)
            # Process rewards and scores
            update_samples_with_rewards(rewards_info, samples_list)
        else:
            # Reward Model
            rewards_list = sum(ray.get(r_refs)[::duplicate_factor], [])
            for i, samples in enumerate(samples_list):
                samples.rewards = rewards_list[i]
                samples.info["reward"] = rewards_list[i]

        assert (
            len(samples_list) == len(action_log_probs_list) == len(base_action_log_probs_list) == len(value_list)
        ), f"len(samples_list): {len(samples_list)}, len(action_log_probs_list): {len(action_log_probs_list)}, len(base_action_log_probs_list): {len(base_action_log_probs_list)}, len(value_list): {len(value_list)}"

        # Process results for each sample
        for i, (samples, action_log_probs, base_action_log_probs, value) in enumerate(
            zip(samples_list, action_log_probs_list, base_action_log_probs_list, value_list)
        ):
            if (self.initial_model_group is not None) and (not args.use_kl_loss):
                kl = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    kl_estimator=self.strategy.args.kl_estimator,
                )
            else:
                kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=device)
            kl_mean = masked_mean(kl, samples.action_mask, dim=-1)

            if not args.use_kl_loss:
                base_action_log_probs = None
            # Update experience with new information
            samples.action_log_probs = action_log_probs
            samples.base_action_log_probs = base_action_log_probs
            samples.values = value
            samples.kl = kl
            samples.info["kl"] = kl_mean

        end_time = time.time()
        duration = end_time - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(f"✨ Experience making completed in {time_str}")
        return samples_list

    @torch.no_grad()
    def compute_advantages_and_returns(
        self, experiences: List[Experience], **kwargs
    ) -> Tuple[List[Experience], List[torch.Tensor]]:
        """
        Process experiences, this can be used to filter out some experiences or do some processing on the rewards.
        Example, use_dynamic_batch
            >>> rewards: [0, 1, 0.5, 1], indices: [1, 2, 0, 3], n_samples_per_prompt: 2
            >>> sorted rewards: [0,5, 0, 1, 1], reward shaping: [0.25, 0.25, 1, 1]
            >>> map back: [0.25, 1, 0.25, 1]
        Output:
        - experiences: List of Experience
        - rewards: List of rewards
        """
        args = self.strategy.args

        # DAPO reward shaping with optional overlong penalty - Apply BEFORE dynamic indices processing
        if args.overlong_buffer_len is not None:
            assert (
                args.generate_max_len >= args.overlong_buffer_len
            ), "generate_max_len must be larger than overlong_buffer_len"
            overlong_buffer_len = args.overlong_buffer_len
            expected_len = args.generate_max_len - overlong_buffer_len
            overlong_penalty_factor = args.overlong_penalty_factor

            # Apply penalty to each experience's rewards based on response length
            for experience in experiences:
                response_lengths = experience.info["response_length"]
                batch_size = len(response_lengths)
                for j in range(batch_size):
                    valid_response_length = response_lengths[j].item()
                    # Cap the exceed_len to overlong_buffer_len to prevent excessive penalty
                    exceed_len = min(valid_response_length - expected_len, overlong_buffer_len)
                    if exceed_len > 0:
                        overlong_penalty = -exceed_len / overlong_buffer_len * overlong_penalty_factor
                        # Apply penalty to the j-th reward in this experience
                        experience.rewards[j] += overlong_penalty

        # get rewards from experiences
        exp_len = [len(experience.index) for experience in experiences]
        # indices is an identity mapping when not using dynamic batch; otherwise, it maps back to the original indices after rearrange samples
        indices = torch.tensor(sum([experience.index for experience in experiences], []))
        raw_rewards = torch.cat([experience.rewards for experience in experiences], dim=0)
        rewards = torch.empty_like(raw_rewards)
        rewards[indices] = raw_rewards  # sorted

        rewards = rewards.reshape(-1, args.n_samples_per_prompt)

        # log group reward std
        if args.n_samples_per_prompt > 1:
            group_reward_stds = (
                rewards.std(-1, keepdim=True).repeat(1, args.n_samples_per_prompt).reshape(-1)[indices].split(exp_len)
            )
            for experience, group_reward_std in zip(experiences, group_reward_stds):
                experience.info["group_reward_std"] = group_reward_std

        # reward shaping
        if args.advantage_estimator == "rloo":
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (args.n_samples_per_prompt - 1)
            rewards = rewards - baseline
        elif args.advantage_estimator in ["reinforce_baseline", "dr_grpo"]:
            # REINFORCE++-baseline and Dr. GRPO removed the `/std` in GRPO as `/ std` is not needed in RL variance reduction theory.
            # And `k3 KL` has a larger variance than `k1 KL` under a categorical distribution.
            rewards = rewards - rewards.mean(-1, keepdim=True)
        elif args.advantage_estimator == "group_norm":
            rewards = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-9)
        rewards = rewards.reshape(-1)[indices].split(exp_len)
        # calculate return and advantages
        for experience, reward in zip(experiences, rewards):
            reward = compute_reward(
                reward,
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                reward_clip_range=args.reward_clip_range,
            )

            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    args.gamma,
                    args.lambd,
                )
            elif self.advantage_estimator in ["reinforce", "rloo", "reinforce_baseline", "group_norm", "dr_grpo"]:
                if args.gamma != 1.0 and self.advantage_estimator in [
                    "rloo",
                    "reinforce_baseline",
                    "group_norm",
                    "dr_grpo",
                ]:
                    logger.warning("gamma is set to 1.0 for rloo, reinforce_baseline, and group_norm")
                    args.gamma = 1.0

                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    args.gamma,
                )
                experience.advantages = deepcopy(experience.returns)
            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")

            # calculate the return info.
            return_sums = reward.sum(dim=-1)
            experience.info["return"] = return_sums
            # remove unnecessary info
            experience.kl = None

        # Normalize advantages across all experiences for GAE, REINFORCE, and REINFORCE-baseline
        if self.args.advantage_estimator in ["gae", "reinforce", "reinforce_baseline"]:
            all_advantages = []
            all_action_masks = []
            for exp in experiences:
                all_advantages.append(exp.advantages.flatten())
                all_action_masks.append(exp.action_mask.flatten())

            advantages_vector = torch.cat(all_advantages, dim=0).float()
            action_masks_vector = torch.cat(all_action_masks, dim=0)
            num_actions = action_masks_vector.sum()

            # mean
            mean = (advantages_vector * action_masks_vector).sum() / num_actions
            # std
            if not self.args.no_advantage_std_norm:
                var = ((advantages_vector - mean).pow(2) * action_masks_vector).sum() / num_actions
                rstd = var.clamp(min=1e-8).rsqrt()
            else:
                rstd = 1

            # Apply normalization to each experience
            for exp in experiences:
                exp.advantages = (exp.advantages - mean) * rstd

        return experiences

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns

    @torch.no_grad()
    def get_cumulative_returns(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that computes advantages and returns from rewards using REINFORCE.
        REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

        Input:
        - rewards: Tensor of shape (batch_size, response_size)
        - action_mask: Tensor of shape (batch_size, response_size), binary mask
        - gamma: discount factor

        Output:
        - returns: Tensor of shape (batch_size, response_size)
        """
        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        if action_mask is not None:
            rewards = action_mask * rewards

        # Calculate returns by accumulating discounted rewards
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return

        return returns
