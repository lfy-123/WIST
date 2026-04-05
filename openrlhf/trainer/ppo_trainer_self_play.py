import os
import time
from abc import ABC
from datetime import timedelta
import ray
import torch
from tqdm import tqdm
from openrlhf.datasets import PromptDataset
from openrlhf.datasets.utils import blending_datasets
from openrlhf.trainer.ppo_utils import AdaptiveKLController, FixedKLController
from openrlhf.trainer.ppo_utils.experience_maker_self_play import RemoteExperienceMaker
from openrlhf.trainer.ppo_utils.replay_buffer import balance_experiences
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.utils import get_tokenizer
import numpy as np
from knowledge_tree.Tree import BasicTree
import random
from typing import Optional
# from openrlhf.prompts.challenger import FREE_PROMPT, FREE_PROMPT_WITH_PATH, MCQ_PROMPT, MCQ_PROMPT_WITH_PATH, MCQ_PROMPT_SIMPLE, MCQ_PROMPT_SIMPLE_WITH_PATH
from openrlhf.prompts.judge import QUESTION_TYPE_JUDGE_PROMPT
from openrlhf.prompts.utils import  parse_json_output, parse_box_output
import json
import glob
import tempfile

logger = init_logger(__name__)

def show_path(path):
    idx = 0
    if path[0].name == "Root":
        idx = 1
    names = []
    for node in path[idx:]:
        names.append(node.name)
    return(' -> '.join(names))


import re
HAS_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")  # CJK unified ideographs
def validate_language(title: str) -> bool:
    if not title or not title.strip():
        print(f"[验证失败] 传入的内容为空或全空白")
        return False
    title = title.strip()

    if HAS_CJK_PATTERN.search(title):
        print(f"[验证失败] 传入的内容包含中文: {title}")
        return False
    return True



class BasePPOTrainer(ABC):
    def __init__(
        self,
        pretrain: str,
        strategy: DeepspeedStrategy,
        actor_model_group: RayActorGroup,
        critic_model_group: RayActorGroup,
        reward_model_group: RayActorGroup,
        reference_model_group: RayActorGroup,
        vllm_engines=None,
        prompt_max_len: int = 120,
        dataloader_pin_memory: bool = True,
        prompt_split: str = "train",
        eval_split: str = "test",
        max_steps: int = 650,
        **generate_kwargs,
    ) -> None:
        super().__init__()

        self.strategy = strategy
        self.args = strategy.args

        self.tokenizer = get_tokenizer(pretrain, None, "left", strategy, use_fast=not self.args.disable_fast_tokenizer)
        self.actor_model_group = actor_model_group
        self.critic_model_group = critic_model_group
        self.reward_model_group = reward_model_group
        self.reference_model_group = reference_model_group
        self.dataloader_pin_memory = dataloader_pin_memory
        self.vllm_engines = vllm_engines

        self.prompt_split = prompt_split
        self.eval_split = eval_split

        self.prompt_max_len = prompt_max_len
        self.generate_kwargs = generate_kwargs

        self.max_epochs = self.args.max_epochs
        self.remote_rm_url = self.args.remote_rm_url
        self.init_kl_coef = self.args.init_kl_coef
        self.kl_target = self.args.kl_target
        self.kl_horizon = self.args.kl_horizon

        self.freezing_actor_steps = getattr(self.args, "freezing_actor_steps", -1)

        # Init dummy variables
        self.prompts_dataloader = None
        self.eval_dataloader = None
        self.max_steps = max_steps

        self.samples_generator = None
        self.experience_maker = None
        self.remote_reward_model = None

        if self.args.agent_func_path:
            from openrlhf.trainer.ppo_utils.experience_maker_async import SamplesGeneratorAsync as SamplesGenerator
        else:
            # from openrlhf.trainer.ppo_utils.experience_maker import SamplesGenerator
            from openrlhf.trainer.ppo_utils.experience_maker_self_play import SamplesGenerator

        self.generator_cls = SamplesGenerator

    def _init_wandb(self):
        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        self.generated_samples_table = None
        if self.strategy.args.use_wandb:
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=self.strategy.args.use_wandb)
            wandb.init(
                entity=self.strategy.args.wandb_org,
                project=self.strategy.args.wandb_project,
                group=self.strategy.args.wandb_group,
                name=self.strategy.args.wandb_run_name,
                config=self.strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/epoch")
            wandb.define_metric("eval/*", step_metric="eval/epoch", step_sync=True)
            self.generated_samples_table = wandb.Table(columns=["global_step", "text", "reward"])

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None:
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, self.strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def fit(self):
        raise NotImplementedError("fit method is not implemented")

    def ppo_train(self, global_steps):
        status = {}

        # triger remote critic model training
        if self.critic_model_group is not None:
            # sync for deepspeed_enable_sleep
            if self.strategy.args.deepspeed_enable_sleep:
                ray.get(self.critic_model_group.async_run_method(method_name="reload_states"))

            critic_status_ref = self.critic_model_group.async_run_method(method_name="fit")

            if self.strategy.args.colocate_all_models or self.strategy.args.deepspeed_enable_sleep:
                status.update(ray.get(critic_status_ref)[0])
            if self.strategy.args.deepspeed_enable_sleep:
                ray.get(self.critic_model_group.async_run_method(method_name="offload_states"))

        # actor model training
        if global_steps > self.freezing_actor_steps:
            if self.strategy.args.deepspeed_enable_sleep:
                ray.get(self.actor_model_group.async_run_method(method_name="reload_states"))

            actor_status_ref = self.actor_model_group.async_run_method(method_name="fit", kl_ctl=self.kl_ctl.value)
            status.update(ray.get(actor_status_ref)[0])

            if self.strategy.args.deepspeed_enable_sleep:
                ray.get(self.actor_model_group.async_run_method(method_name="offload_states"))

            # 4. broadcast weights to vllm engines
            if self.vllm_engines is not None:
                self._broadcast_to_vllm()

        # 5. wait remote critic model training done
        if self.critic_model_group and not self.strategy.args.colocate_all_models:
            status.update(ray.get(critic_status_ref)[0])

        return status

    def _broadcast_to_vllm(self):
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        ray.get(self.actor_model_group.async_run_method(method_name="broadcast_to_vllm"))

        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None:
                # Add generated samples to wandb using Table
                if "generated_samples" in logs_dict:
                    # https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
                    new_table = self._wandb.Table(
                        columns=self.generated_samples_table.columns, data=self.generated_samples_table.data
                    )
                    new_table.add_data(global_step, *logs_dict.pop("generated_samples"))
                    self.generated_samples_table = new_table
                    self._wandb.log({"train/generated_samples": new_table})
                logs = {
                    "train/%s" % k: v
                    for k, v in {
                        **logs_dict,
                        "global_step": global_step,
                    }.items()
                }
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None:
                for k, v in logs_dict.items():
                    if k == "generated_samples":
                        # Record generated samples in TensorBoard using simple text format
                        text, reward = v
                        formatted_text = f"Sample:\n{text}\n\nReward: {reward:.4f}"
                        self._tensorboard.add_text("train/generated_samples", formatted_text, global_step)
                    else:
                        self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # TODO: Add evaluation mechanism for PPO
        if global_step % args.eval_steps == 0 and self.eval_dataloader and len(self.eval_dataloader) > 0:
            self.evaluate(self.eval_dataloader, global_step, args.eval_temperature, args.eval_n_samples_per_prompt)
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity/others on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            ref = self.actor_model_group.async_run_method(
                method_name="save_checkpoint", tag=tag, client_states=client_states
            )
            if self.critic_model_group is not None:
                ref.extend(self.critic_model_group.async_run_method(method_name="save_checkpoint", tag=tag))
            ray.get(ref)

    def evaluate(self, eval_dataloader, global_step, temperature=0.6, n_samples_per_prompt=1):
        """Evaluate model performance on eval dataset.

        Args:
            eval_dataloader: DataLoader containing evaluation prompts, labels and data sources
            global_step: Current training step for logging
            n_samples_per_prompt: Number of samples to generate per prompt for pass@k calculation
        """
        start_time = time.time()
        logger.info(f"⏰ Evaluation start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # vLLM wakeup when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        with torch.no_grad():
            # First collect all prompts and labels
            all_prompts = []
            all_labels = []
            prompt_to_datasource = {}  # Dictionary to store mapping between prompts and their data sources

            for datasources, prompts, labels in eval_dataloader:
                all_prompts.extend(prompts)
                all_labels.extend(labels)
                # Create mapping for each prompt to its corresponding data source
                for prompt, datasource in zip(prompts, datasources):
                    prompt_to_datasource[prompt] = datasource

            # Generate samples and calculate rewards
            generate_kwargs = self.generate_kwargs.copy()
            generate_kwargs["temperature"] = temperature
            generate_kwargs["n_samples_per_prompt"] = n_samples_per_prompt
            samples_list = self.samples_generator.generate_samples(
                all_prompts, all_labels, remote_reward_model=self.remote_reward_model, **generate_kwargs
            )

            # duplicate prompts and labels for each sample
            all_prompts = sum([s.prompts for s in samples_list], [])
            all_labels = sum([s.labels for s in samples_list], [])

            # Get rewards from samples, such as agent rewards or remote reward models
            rewards_list = []
            for samples in samples_list:
                rewards_list.append(samples.rewards)
            # Reshape rewards to (num_prompts, n_samples_per_prompt)
            rewards = torch.tensor(rewards_list).reshape(-1, n_samples_per_prompt)

            # Collect local statistics for each data source
            global_metrics = {}  # {datasource: {"pass{n_samples_per_prompt}": 0, "pass1": 0, "count": 0}}

            # Process rewards in chunks of n_samples_per_prompt
            num_prompts = len(all_prompts) // n_samples_per_prompt
            for i in range(num_prompts):
                # Get the original prompt (first one in the chunk)
                original_prompt = all_prompts[i * n_samples_per_prompt]
                datasource = prompt_to_datasource[original_prompt]  # Get corresponding data source using the mapping

                if datasource not in global_metrics:
                    global_metrics[datasource] = {f"pass{n_samples_per_prompt}": 0, "pass1": 0, "count": 0}

                # Get rewards for this chunk
                chunk_rewards = rewards[i]

                # Calculate pass@k and pass@1
                if n_samples_per_prompt > 1:
                    global_metrics[datasource][f"pass{n_samples_per_prompt}"] += chunk_rewards.max().float().item()
                global_metrics[datasource]["pass1"] += chunk_rewards.mean().float().item()
                global_metrics[datasource]["count"] += 1

            # Calculate global averages
            logs = {}
            for datasource, metrics in global_metrics.items():
                logs[f"eval_{datasource}_pass{n_samples_per_prompt}"] = (
                    metrics[f"pass{n_samples_per_prompt}"] / metrics["count"]
                )
                logs[f"eval_{datasource}_pass1"] = metrics["pass1"] / metrics["count"]

            # Log to wandb/tensorboard
            if self._wandb is not None:
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": global_step}.items()}
                self._wandb.log(logs)
            elif self._tensorboard is not None:
                for k, v in logs.items():
                    self._tensorboard.add_scalar(f"eval/{k}", v, global_step)

        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

        end_time = time.time()
        duration = end_time - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(f"✨ Evaluation completed in {time_str}, global_step {global_step}, eval_metrics: {logs}")

    def prepare_datasets(self):
        args = self.args
        strategy = self.strategy

        # prepare datasets
        train_data = blending_datasets(
            args.prompt_data,
            args.prompt_data_probs,
            strategy,
            args.seed,
            max_count=args.max_samples,
            dataset_split=self.prompt_split,
        )

        # Create train dataset
        train_data = train_data.select(range(min(args.max_samples, len(train_data))))
        prompts_dataset = PromptDataset(train_data, self.tokenizer, strategy, input_template=args.input_template)
        prompts_dataloader = strategy.setup_dataloader(
            prompts_dataset,
            args.vllm_generate_batch_size,
            True,
            True,
        )

        # Create eval dataset if eval data exists
        if getattr(args, "eval_dataset", None):
            eval_data = blending_datasets(
                args.eval_dataset,
                None,  # No probability sampling for eval datasets
                strategy,
                dataset_split=self.eval_split,
            )
            eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))
            eval_dataset = PromptDataset(eval_data, self.tokenizer, strategy, input_template=args.input_template)
            eval_dataloader = strategy.setup_dataloader(eval_dataset, 1, True, False)
        else:
            eval_dataloader = None

        self.prompts_dataloader = prompts_dataloader
        self.eval_dataloader = eval_dataloader
        # self.max_steps = (
        #     len(prompts_dataset)
        #     * args.n_samples_per_prompt
        #     // args.train_batch_size
        #     * args.num_episodes
        #     * args.max_epochs
        # )
        
    def get_max_steps(self):
        return self.max_steps


@ray.remote
class PPOTrainer(BasePPOTrainer):
    """
    Trainer for Proximal Policy Optimization (PPO) / REINFORCE++ / GRPO / RLOO and their variants.
    Single Controller with Multiple ActorGroups
    """

    def __init__(
        self,
        pretrain: str,
        strategy: DeepspeedStrategy,
        tree_kwargs: dict,
        actor_model_group: RayActorGroup,
        critic_model_group: RayActorGroup,
        reward_model_group: RayActorGroup,
        reference_model_group: RayActorGroup,
        vllm_engines=None,
        prompt_max_len: int = 120,
        dataloader_pin_memory: bool = True,
        prompt_split: str = "train",
        eval_split: str = "test",
        max_steps: int = 650,
        only_sample: bool = False,
        clean_web_stop_steps: int = 10,
        start_steps_judge_type: int = 10,
        **generate_kwargs,
    ) -> None:
        super().__init__(
            pretrain,
            strategy,
            actor_model_group,
            critic_model_group,
            reward_model_group,
            reference_model_group,
            vllm_engines,
            prompt_max_len,
            dataloader_pin_memory,
            prompt_split,
            eval_split,
            max_steps,
            **generate_kwargs,
        )

        if self.kl_target:
            self.kl_ctl = AdaptiveKLController(self.init_kl_coef, self.kl_target, self.kl_horizon)
        else:
            self.kl_ctl = FixedKLController(self.init_kl_coef)

        if self.args.remote_rm_url and not self.args.remote_rm_url[0] == "agent":
            from openrlhf.utils.remote_rm_utils import RemoteRewardModel
            self.remote_reward_model = RemoteRewardModel.remote(self.args, self.remote_rm_url)

        self.samples_generator = self.generator_cls(
            self.vllm_engines,
            self.strategy,
            self.tokenizer,
            self.prompt_max_len,
        )

        self.experience_maker = RemoteExperienceMaker(
            self.actor_model_group,
            self.critic_model_group,
            self.reward_model_group,
            self.reference_model_group,
            self.kl_ctl,
            self.strategy,
            self.tokenizer,
            remote_reward_model=self.remote_reward_model,
        )

        # self.prepare_datasets()
        self._init_wandb()

        # get eval and save steps
        if self.args.eval_steps == -1:
            self.args.eval_steps = float("inf")  # do not evaluate
        if self.args.save_steps == -1:
            self.args.save_steps = float("inf")  # do not save ckpt
        
        self.tree = BasicTree(**tree_kwargs)
        self.tree.tokenizer = self.tokenizer
        self.only_sample = only_sample
        self.paths_ckpt_path_dir = os.path.join(self.args.ckpt_path, "sampled_paths")
        self.challenger_output_log_path_dir = os.path.join(self.args.ckpt_path, "challenger_output")
        self.judge_output_log_path_dir = os.path.join(self.args.ckpt_path, "judge_output")
        self.experience_log_path_dir = os.path.join(self.args.ckpt_path, "experience_logs")
        
        os.makedirs(self.paths_ckpt_path_dir, exist_ok=True)
        os.makedirs(self.challenger_output_log_path_dir, exist_ok=True)
        os.makedirs(self.judge_output_log_path_dir, exist_ok=True)
        os.makedirs(self.experience_log_path_dir, exist_ok=True)

        self.instruction_len = self.args.instruction_len  # 2048  
        self.truncate_size = self.prompt_max_len - self.instruction_len  
        self.clean_web_stop_steps = clean_web_stop_steps
        self.start_steps_judge_type = start_steps_judge_type
        self.clean_web = self.args.clean_web
        
    def sample_from_tree(self, batch_size: int, models, sampled_paths, specified_domain: Optional[str] = None) -> list:
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
            batch_vllm_engine_call(self.vllm_engines, "wake_up")
        
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)
        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template
        
        if len(sampled_paths) != 0:
            paths = []
            for path in sampled_paths:
                node_path = self.tree.get_nodes_by_path(path)
                if len(node_path) == self.tree.max_levels + 1:
                    paths.append(node_path)
        else:
            paths = []

        max_rounds = batch_size * 3   
        rounds = 0
        while len(paths) < batch_size and rounds < max_rounds:
            rounds += 1
            remaining = batch_size - len(paths)
            paths.extend(self.tree.sample_paths(
                batch_size=remaining,
                models=models,
                only_sample=self.only_sample,              
                specified_domain=None,
                paths_ckpt_path=self.paths_ckpt_path,  
                sampled_paths=sampled_paths,  
            ))
            self.tree.update_web_corpus()
        
        if self.args.title_selection_mode == "random":
            contents, paths = self._get_contents_from_paths(paths)
        else:
            contents, paths = self._get_contents_from_paths_by_sequence(paths)
            
        while len(contents) < batch_size:
            print(f"[sample_from_tree] Warning: Expected batch_size={batch_size}, actually collected {len(contents)} valid samples.")
            remaining = batch_size - len(contents)
            if not paths:
                break
            k = min(remaining, len(paths))
            random_paths = random.sample(paths, k)
            
            if self.args.title_selection_mode == "random":
                remaining_contents, remaining_paths = self._get_contents_from_paths(random_paths)
            else:
                remaining_contents, remaining_paths = self._get_contents_from_paths_by_sequence(random_paths)
            contents.extend(remaining_contents)
            paths.extend(remaining_paths)
        # vLLM offload when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")
        paths_str = [show_path(path) for path in paths]
        assert len(contents) == len(paths_str)
        return contents, paths_str    
    
    def _get_contents_from_paths_by_sequence(self, paths) -> str:
        def _atomic_save_json(path: str, obj: dict) -> None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(prefix="path_title_cursor_", suffix=".json", dir=os.path.dirname(path))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(obj, f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, path)
            except Exception:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                raise

        def _load_state(path: str) -> dict:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], dict):
                    return obj["data"]
                if isinstance(obj, dict):
                    return obj
                return {}
            except Exception:
                return {}

        def _latest_state_file(state_dir: str, episode: int) -> str:
            pattern = os.path.join(state_dir, f"path_title_cursor_ep{episode}_step*.json")
            files = glob.glob(pattern)
            if not files:
                return ""
            def _step_num(fp: str) -> int:
                base = os.path.basename(fp)
                try:
                    n = base.split("_step", 1)[1].rsplit(".json", 1)[0]
                    return int(n)
                except Exception:
                    return -1
            files.sort(key=_step_num)
            return files[-1]

        episode = self.episode
        step = self.step
        state_dir = os.path.join(self.args.ckpt_path, "path_title_cursor")
        os.makedirs(state_dir, exist_ok=True)

        prev_step = step - 1
        prev_path = os.path.join(state_dir, f"path_title_cursor_ep{episode}_step{prev_step}.json") if prev_step >= 0 else ""
        cur_path = os.path.join(state_dir, f"path_title_cursor_ep{episode}_step{step}.json")

        if prev_path and os.path.exists(prev_path):
            cursor_state = _load_state(prev_path)
        else:
            latest = _latest_state_file(state_dir, episode)
            cursor_state = _load_state(latest) if latest else {}

        contents = []
        delete_idx = []
        updated_any = False

        for idx, path in enumerate(paths):
            path_key = show_path(path)
            try:
                all_titles = path[-1].wiki_titles
            except Exception as e:
                print(f"[_get_contents_from_paths] Error accessing wiki_titles for path {path_key}: {e}")
                delete_idx.append(idx)
                continue
            if not all_titles:
                print(f"[_get_contents_from_paths] Warning: Path '{path_key}' has empty wiki_titles.")
                delete_idx.append(idx)
                continue
            print(f"********* The length before similarity filtering is: {len(all_titles)}")
            valid_pairs = []
            for title in all_titles:
                if not title:
                    continue
                title = str(title).strip()
                items = self.tree.web_corpus.get(title, ("", ""))
                content = items[1] if items[1] else items[0]
                if content and validate_language(title) and validate_language(content[:1000]):
                    valid_pairs.append((title, content))
            if not valid_pairs:
                print(f"[_get_contents_from_paths] Warning: Path '{path_key}' has no valid content in web_corpus.")
                delete_idx.append(idx)
                continue
            print(f"********* The length after similarity filtering is: {len(valid_pairs)}")
            cursor = int(cursor_state.get(path_key, 0) or 0)
            chosen_idx = cursor % len(valid_pairs)
            chosen_title, content = valid_pairs[chosen_idx]

            cursor_state[path_key] = cursor + 1
            updated_any = True
            encoded = self.tokenizer(
                content,
                add_special_tokens=False,
                return_attention_mask=False,
            )
            all_ids = encoded["input_ids"]
            n_tokens = len(all_ids)
            if n_tokens > self.truncate_size:
                truncated_ids = all_ids[: self.truncate_size]
                content = self.tokenizer.decode(truncated_ids, skip_special_tokens=True)
                print(f"[sample_from_tree] Warning: sampled text is too long, exceeds the maximum length of the corpus {self.truncate_size}, has been truncated")
            contents.append(content)

        if delete_idx:
            for index in sorted(delete_idx, reverse=True):
                del paths[index]
        if updated_any:
            payload = {
                "meta": {
                    "episode": episode,
                    "step": step,
                    "timestamp": int(time.time()),
                    "total_paths_seen": len(paths),
                    "total_contents_returned": len(contents),
                    "source_prev_file": prev_path if (prev_path and os.path.exists(prev_path)) else None,
                },
                "data": cursor_state,
            }
            try:
                _atomic_save_json(cur_path, payload)
            except Exception as e:
                print(f"[_get_contents_from_paths] Warning: failed to save cursor_state to {cur_path}: {e}")
        return contents, paths
    
    def _get_contents_from_paths(self, paths) -> str:
        contents = []
        delete_idx = []
        for idx, path in enumerate(paths):
            try:
                all_titles = path[-1].wiki_titles
            except Exception as e:
                print(f"[_get_contents_from_paths] Error accessing wiki_titles for path {show_path(path)}': {e}")
                delete_idx.append(idx)
                continue
            print(f"********* The length before similarity filtering is: {len(all_titles)}")
            all_contents = []
            for title in all_titles:
                items = self.tree.web_corpus.get(title, ("", ""))
                content = items[1] if items[1] else items[0]
                if content and validate_language(title) and validate_language(content[:1000]):
                    all_contents.append(content)
            if not all_contents:
                print(f"[_get_contents_from_paths] Warning: Path '{show_path(path)}' has no valid content in web_corpus.")
                delete_idx.append(idx)
                continue
            print(f"********* The length after similarity filtering is: {len(all_contents)}")
            content = random.choice(all_contents)
            encoded = self.tokenizer(
                content,
                add_special_tokens=False,
                return_attention_mask=False,
            )
            all_ids = encoded["input_ids"]
            n_tokens = len(all_ids)
            if n_tokens > self.truncate_size:
                truncated_ids = all_ids[: self.truncate_size]
                content = self.tokenizer.decode(truncated_ids, skip_special_tokens=True)
                print(f"[sample_from_tree] Warning: sampled text is too long, exceeds the maximum length of the corpus {self.truncate_size}, has been truncated")
            contents.append(content)
        if delete_idx:
            for index in sorted(delete_idx, reverse=True):
                del paths[index]
        return contents, paths
    
    def fit(
        self,
    ) -> None:
        args = self.args
        # broadcast init checkpoint to vllm
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path):
            checkpoint_states = ray.get(self.actor_model_group.async_run_method(method_name="get_checkpoint_states"))[
                0
            ]
            logger.info(f"checkpoint_states: {checkpoint_states}")
            self._broadcast_to_vllm()
            rng_state = checkpoint_states.get("rng_state", None)
            if rng_state is not None:
                logger.info("[fit] Restore RNG state from checkpoint.")
                random.setstate(rng_state["python"])
                np.random.set_state(rng_state["numpy"])
                torch.random.set_rng_state(rng_state["torch_cpu"])
                if torch.cuda.is_available() and "torch_cuda" in rng_state:
                    torch.cuda.random.set_rng_state(rng_state["torch_cuda"])
                self.tree.set_rng_state(rng_state)
            else:
                logger.info("[fit] No rng_state in checkpoint, fallback to seeding by args.seed")
                random.seed(args.seed)
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(args.seed)
        else:
            checkpoint_states = {"global_step": 0, "episode": 0, "data_loader_state_dict": {}}
            logger.info("[fit] Init RNG state from args.seed")
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
        # Restore step and start_epoch
        steps = checkpoint_states["global_step"] + 1
        episode = checkpoint_states["episode"]

        for episode in range(episode, args.num_episodes):
            self.episode = episode
            pbar = tqdm(
                range(self.max_steps),
                desc=f"Episode [{episode + 1}/{args.num_episodes}]",
                disable=False,
                initial=steps,
            )
            filtered_samples = []
            number_of_samples = 0
            print(f"[PPOTrainer fit] Starting from step: {steps}")
            for step_idx in range(steps, self.max_steps + 1):
                if self.args.explore_strategy == "random_reward" and step_idx >= 50:  
                    self.tree.explore_strategy = "reward"
                self.step = step_idx
                start_time = time.time()

                self.paths_ckpt_path = os.path.join(self.paths_ckpt_path_dir, f"sampled_paths_ep{episode}_step{step_idx}.json")
                self.challenger_output_log_path = os.path.join(self.challenger_output_log_path_dir, f"challenger_output_ep{episode}_step{step_idx}.log")
                self.judge_output_log_path = os.path.join(self.judge_output_log_path_dir, f"judge_output_ep{episode}_step{step_idx}.log")
                self.experience_log_path = os.path.join(self.experience_log_path_dir, f"experience_ep{episode}_step{steps}.json")
                self.generation_log_path = os.path.join(self.experience_log_path_dir, f"generation_ep{episode}_step{steps}.json")
                
                sampled_paths = []
                if os.path.exists(self.paths_ckpt_path):
                    with open(self.paths_ckpt_path, "r") as f:
                        try:
                            sampled_paths = json.load(f)
                        except Exception as e:
                            print(f"[fit] Error loading sampled_paths from {self.paths_ckpt_path}: {e}")

                if step_idx > self.clean_web_stop_steps and self.clean_web:
                    self.tree.clean_web = True
                else:
                    self.tree.clean_web = False
                
                if step_idx > self.start_steps_judge_type:
                    self.samples_generator.judge_question_type = True
                else:
                    self.samples_generator.judge_question_type = False
                    
                prompts, paths = self.sample_from_tree(args.rollout_batch_size, self.vllm_engines, sampled_paths)  
                
                mid_time_1 = time.time()
                print(f"[fit Time] Generated {len(prompts)} challenger prompts using {mid_time_1 - start_time:.2f} seconds.")
                
                if steps <= self.args.cold_start_steps:
                    path_reward_pairs = {}
                    challenger_rollout_samples, challenger_paths = self.samples_generator.generate_samples(
                        prompts, None, role="challenger", paths=paths, cold_start=True,
                        challenger_output_log_path=self.challenger_output_log_path, 
                        remote_reward_model=None, **self.generate_kwargs
                    )
                    challenger_prompts, challenger_labels = [], []
                    n_samples_per_prompt = self.generate_kwargs.get(
                        "n_samples_per_prompt",
                        getattr(self.strategy.args, "n_samples_per_prompt", 1),
                    )
                    num_rollouts = len(challenger_rollout_samples)
                    assert len(challenger_paths) == num_rollouts, "The length of paths should be the same as the number of samples."
                    assert num_rollouts % n_samples_per_prompt == 0, \
                        f"len(challenger_rollout_samples)={num_rollouts} cannot be divided by n_samples_per_prompt={n_samples_per_prompt}"
                    num_groups = num_rollouts // n_samples_per_prompt
                    
                    for g in range(num_groups):
                        start = g * n_samples_per_prompt
                        end = start + n_samples_per_prompt
                        group_samples = challenger_rollout_samples[start:end]
                        group_path = challenger_paths[start]
                        valid_qas = []
                        group_rewards = []
                        for rollout_sample in group_samples:
                            questions = getattr(rollout_sample, "questions", None)
                            answers = getattr(rollout_sample, "answers", None)
                            if not questions or not answers:
                                continue

                            question = str(questions[0]).strip()
                            answer = str(answers[0]).strip()
                            if question and answer:
                                valid_qas.append((question, answer))
                            
                            group_rewards.append(rollout_sample.rewards.item())

                        max_r = max(group_rewards)
                        if group_path not in path_reward_pairs:
                            path_reward_pairs[group_path] = [max_r]
                        else:
                            path_reward_pairs[group_path].append(max_r) 
                        
                        assert len(valid_qas) > 0, f"[fit] group {g} has no valid QA, need check!!!"
                        chosen_q, chosen_a = random.choice(valid_qas)
                        challenger_prompts.append(chosen_q)
                        challenger_labels.append(chosen_a)

                    reasoner_rollout_samples = []
                    if challenger_prompts:
                        reasoner_rollout_samples, _ = self.samples_generator.generate_samples(
                            challenger_prompts, challenger_labels, role="reasoner", cold_start=True, remote_reward_model=None, **self.generate_kwargs)
                    generation_logs = None
                    
                    if self.tree.explore_strategy == "reward":
                        self.tree.update_with_reward_feedback(path_reward_pairs)
                else:
                    challenger_rollout_samples, reasoner_rollout_samples, path_reward_pairs, generation_logs = self.samples_generator.generate_samples(
                        prompts,
                        None,
                        role="challenger",
                        paths=paths,
                        remote_reward_model=None,
                        return_reasoner_experiences=True,
                        return_generation_logs=True,
                        **self.generate_kwargs
                    )
                    
                    if self.tree.explore_strategy == "reward":
                        self.tree.update_with_reward_feedback(path_reward_pairs)
                    
                if generation_logs:
                    with open(self.generation_log_path, "w", encoding="utf-8") as f:
                        json.dump(generation_logs, f, ensure_ascii=False, indent=2)  

                pass_rate = None
                if self.args.dynamic_filtering:
                    challenger_number_of_samples += len(challenger_rollout_samples)
                    for i in range(0, len(challenger_rollout_samples), self.args.n_samples_per_prompt):
                        batch_samples = challenger_rollout_samples[i : i + self.args.n_samples_per_prompt]
                        if len(batch_samples) < self.args.n_samples_per_prompt:
                            continue

                        reward_list = [sample.scores[0].item() for sample in batch_samples]
                        avg_reward = sum(reward_list) / len(batch_samples)
                        
                        if any(reward > 0 for reward in reward_list) and avg_reward < 1.0:
                            challenger_filtered_samples.extend(batch_samples)

                    pass_rate = len(challenger_filtered_samples) / challenger_number_of_samples * 100
                    logger.info(
                        f"Dynamic filtering challenger samples pass rate: {pass_rate:.2f}% ({len(challenger_filtered_samples)}/{challenger_number_of_samples})"
                    )
                    challenger_rollout_samples = challenger_filtered_samples[: self.args.rollout_batch_size * self.args.n_samples_per_prompt]
                    challenger_filtered_samples = []
                    challenger_number_of_samples = 0
                
                # dynamic filtering 
                pass_rate = None
                if self.args.dynamic_filtering and reasoner_rollout_samples:  
                    reasoner_number_of_samples += len(reasoner_rollout_samples)
                    # Group individual samples into batches of n_samples size
                    for i in range(0, len(reasoner_rollout_samples), self.args.n_samples_per_prompt):
                        batch_samples = reasoner_rollout_samples[i : i + self.args.n_samples_per_prompt]
                        if len(batch_samples) < self.args.n_samples_per_prompt:
                            continue
                        reward_list = [sample.scores[0].item() for sample in batch_samples]
                        avg_reward = sum(reward_list) / len(batch_samples)

                        min_reward, max_reward = self.args.dynamic_filtering_reward_range
                        if min_reward + 1e-6 < avg_reward < max_reward - 1e-6:
                            reasoner_filtered_samples.extend(batch_samples)

                    pass_rate = len(reasoner_filtered_samples) / reasoner_number_of_samples * 100
                    logger.info(
                        f"Dynamic filtering reasoner pass rate: {pass_rate:.2f}% ({len(reasoner_filtered_samples)}/{reasoner_number_of_samples})"
                    )
                    reasoner_rollout_samples = reasoner_filtered_samples[: self.args.rollout_batch_size * self.args.n_samples_per_prompt]
                    reasoner_filtered_samples = []
                    reasoner_number_of_samples = 0
                
                if self.args.dynamic_filtering:
                    print(f"[fit] after filtering: {len(challenger_rollout_samples)} challenger samples, {len(reasoner_rollout_samples)} reasoner samples")
                
                mid_time_2 = time.time()
                print(f"[fit Time] Making {len(challenger_rollout_samples)} exp for challenger and {len(reasoner_rollout_samples)} exp for reasoner {mid_time_2 - mid_time_1:.2f} seconds.")
                pbar.update()
                
                exp_c = self.experience_maker.make_experience_batch(challenger_rollout_samples)
                exp_r = self.experience_maker.make_experience_batch(reasoner_rollout_samples)
                for exp in exp_c:
                    exp.info["role_id"] = torch.tensor([0], dtype=torch.long)  # challenger
                for exp in exp_r:
                    exp.info["role_id"] = torch.tensor([1], dtype=torch.long)  # reasoner

                experiences = exp_c + exp_r
                
                try:
                    records = []
                    for exp in experiences:
                        records.append({
                            "prompts": exp.prompts,
                            "labels": exp.labels,
                            "question": exp.questions,
                            "answer": exp.answers,
                            "question_type": exp.question_type,
                            # "reward": exp.rewards.detach().cpu().tolist()[0],
                            # "advantage": exp.advantages.detach().cpu().tolist()[0],
                        })
                    with open(self.experience_log_path, "w", encoding="utf-8") as f:
                        json.dump(records, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"[ERROR] save experience_log failed: {e}")
                    
                
                sample0 = self.tokenizer.batch_decode(
                    experiences[0].sequences[0].unsqueeze(0), skip_special_tokens=True
                )
                print(sample0)

                # balance experiences across dp
                if args.use_dynamic_batch:
                    experiences = balance_experiences(experiences, args)

                refs = self.actor_model_group.async_run_method_batch(method_name="append", experience=experiences)
                if self.critic_model_group is not None:
                    refs.extend(
                        self.critic_model_group.async_run_method_batch(method_name="append", experience=experiences)
                    )
                ray.get(refs)

                status = self.ppo_train(steps)
                
                mid_time_3 = time.time()
                print(f"[fit Time] Training Policy for Two Role challenger and reasoner using {mid_time_3 - mid_time_2:.2f} seconds.")
                

                if "kl" in status:
                    self.kl_ctl.update(status["kl"], args.rollout_batch_size * args.n_samples_per_prompt)

                # Add generated samples to status dictionary
                # if self.args.dynamic_filtering:
                #     status["dynamic_filtering_pass_rate"] = pass_rate
                logger.info(f"✨ Global step {steps}: {status}")
                status["generated_samples"] = [sample0[0], experiences[0].info["reward"][0]]

                # logs/checkpoints
                client_states = {
                    "global_step": steps,
                    "episode": episode,
                    # "data_loader_state_dict": self.prompts_dataloader.state_dict(),
                    "rng_state": {
                        "python": random.getstate(),
                        "numpy": np.random.get_state(),
                        "torch_cpu": torch.random.get_rng_state(),
                        "torch_cuda": torch.cuda.random.get_rng_state_all()[0]
                        if torch.cuda.is_available()
                        else None,
                    },
                }
                self.save_logs_and_checkpoints(args, steps, pbar, status, client_states)

                steps = steps + 1
                
                end_time = time.time()
                print(f"[fit Time] Episode {episode} Step {step_idx} completed in {end_time - start_time:.2f} seconds.")

        if self._wandb is not None:
            self._wandb.finish()
        if self._tensorboard is not None:
            self._tensorboard.close()
