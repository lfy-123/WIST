import argparse
from datetime import datetime
import ray
from ray.util.placement_group import placement_group
from openrlhf.trainer.ray import create_vllm_engines
from openrlhf.trainer.ray.launcher import (
    RayActorGroup,
    ReferenceModelActor,
)
from openrlhf.trainer.ray.ppo_actor import PolicyModelActor
from openrlhf.trainer.ray.ppo_critic import CriticModelActor
from openrlhf.utils import get_strategy

def train(args):
    if not ray.is_initialized():
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}})

    strategy = get_strategy(args)
    strategy.print(args)

    pg = None
    if args.colocate_actor_ref or args.colocate_all_models:
        if args.init_kl_coef > 0:
            assert (
                args.actor_num_nodes == args.ref_num_nodes
                and args.actor_num_gpus_per_node == args.ref_num_gpus_per_node
            ), f"num_nodes and num_gpus_per_node must be the same when colocate actor and ref model."

        bundles = [{"GPU": 1, "CPU": 1} for _ in range(args.actor_num_nodes * args.actor_num_gpus_per_node)]
        pg = placement_group(bundles, strategy="PACK")
        ray.get(pg.ready())

    vllm_engines = None
    if args.vllm_num_engines is not None and args.vllm_num_engines > 0:
        max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
        if args.colocate_all_models and not args.async_train:
            assert (
                args.actor_num_nodes * args.actor_num_gpus_per_node
                == args.vllm_num_engines * args.vllm_tensor_parallel_size
            ), (
                f"actor_num_nodes * actor_num_gpus_per_node must be equal to "
                f"vllm_num_engines * vllm_tensor_parallel_size, got {args.actor_num_nodes * args.actor_num_gpus_per_node} "
                f"and {args.vllm_num_engines * args.vllm_tensor_parallel_size}"
            )

        if args.agent_func_path:
            from openrlhf.trainer.ray.vllm_engine_async import LLMRayActorAsync as LLMRayActor
        else:
            from openrlhf.trainer.ray.vllm_engine import LLMRayActor

        vllm_engines = create_vllm_engines(
            args.vllm_num_engines,
            args.vllm_tensor_parallel_size,
            args.pretrain,
            args.seed,
            args.full_determinism,
            args.enable_prefix_caching,
            args.enforce_eager,
            max_len,
            pg if args.colocate_all_models and not args.async_train else None,
            args.vllm_gpu_memory_utilization,
            args.vllm_enable_sleep,
            LLMRayActor,
            "processed_logprobs" if args.enable_vllm_is_correction else None,
            args.agent_func_path,
        )

    actor_model = RayActorGroup(
        args.actor_num_nodes,
        args.actor_num_gpus_per_node,
        PolicyModelActor,
        pg=pg,
        num_gpus_per_actor=0.2 if pg else 1,
        duplicate_actors=args.ring_attn_size * args.ds_tensor_parallel_size,
    )

    if args.init_kl_coef <= 0:
        ref_model = None
    else:
        ref_model = RayActorGroup(
            args.ref_num_nodes,
            args.ref_num_gpus_per_node,
            ReferenceModelActor,
            pg=pg,
            num_gpus_per_actor=0.2 if pg else 1,
            duplicate_actors=args.ring_attn_size * args.ds_tensor_parallel_size,
        )

    if not args.colocate_all_models:
        pg = None

    if args.critic_pretrain and args.colocate_critic_reward:
        assert (
            args.critic_num_nodes == args.reward_num_nodes
            and args.critic_num_gpus_per_node == args.reward_num_gpus_per_node
        ), f"num_nodes and num_gpus_per_node must be the same when colocate critic and reward model."

        bundles = [{"GPU": 1, "CPU": 1} for _ in range(args.critic_num_nodes * args.critic_num_gpus_per_node)]
        pg = placement_group(bundles, strategy="PACK")
        ray.get(pg.ready())

    if args.critic_pretrain:
        critic_model = RayActorGroup(
            args.critic_num_nodes,
            args.critic_num_gpus_per_node,
            CriticModelActor,
            pg=pg,
            num_gpus_per_actor=0.2 if pg else 1,
            duplicate_actors=args.ring_attn_size * args.ds_tensor_parallel_size,
        )
    else:
        critic_model = None

    from openrlhf.trainer.ppo_trainer_self_play import PPOTrainer

    domain_tag = args.multi_domain_version or "-".join(args.specified_domain).replace(" ", "_")

    tree_kwargs = dict(
        max_levels=args.max_levels,
        knowledge_tree_path=f"{args.storage_path}/{args.model_name}_{domain_tag}_knowledge_tree.json",
        basic_tree_save_path=f"{args.storage_path}/{args.model_name}_{domain_tag}_basic_tree.json",
        basic_tree_load_path=f"{args.storage_path}/{args.model_name}_{domain_tag}_basic_tree.json",
        window_size=args.tree_window_size,
        thr=0.1,
        explore_strategy=args.explore_strategy,
        fixed_domains=args.specified_domain,
        fixed_domains_ratio=args.specified_domains_ratio,
        seed=args.seed,
        web_corpus_path=args.web_corpus_path,
        max_nodes_nums_once={"1": 4, "2": 4, "3": 4, "4": 8},
        times_for_nodes_limit=20,
        wait_interval=1,
        tokenizer=None,
        clean_prompt_max_len=16384,
        clean_max_new_tokens=16384,
        corpus_chunk_tokens=8192,
        corpus_overlap_tokens=1024,
        clean_web=args.clean_web,
        prompt_version=args.tree_prompt_version,
        specified_backbone=SPECIFIED_BACKBONE,
        select_top_k_nodes=args.select_top_k_nodes,
        min_count=args.select_min_count,
        disable_thinking_mode_or_not=args.disable_thinking_mode_or_not,
    )

    reward_model = None
    ppo_trainer = PPOTrainer.remote(
        args.pretrain,
        strategy,
        tree_kwargs,
        actor_model,
        critic_model,
        reward_model,
        ref_model,
        vllm_engines,
        prompt_split=args.prompt_split,
        eval_split=args.eval_split,
        max_steps=args.max_steps,
        only_sample=args.only_sample,
        clean_web_stop_steps=args.clean_web_stop_steps,
        question_type_judge_start_step=args.question_type_judge_start_step,
        do_sample=True,
        prompt_max_len=args.prompt_max_len,
        max_new_tokens=args.generate_max_len,
        max_length=args.max_len,
        temperature=args.temperature,
        top_p=args.top_p,
        n_samples_per_prompt=args.n_samples_per_prompt,
        num_reasoner_samples=args.num_reasoner_samples,
        challenger_max_resample_attempts=args.challenger_max_resample_attempts,
    )

    max_steps = ray.get(ppo_trainer.get_max_steps.remote())

    refs = []
    if ref_model is not None:
        refs.extend(ref_model.async_init_model_from_pretrained(strategy, args.pretrain))
    refs.extend(actor_model.async_init_model_from_pretrained(strategy, args.pretrain, max_steps, vllm_engines))
    ray.get(refs)

    if args.critic_pretrain:
        refs.extend(critic_model.async_init_model_from_pretrained(strategy, args.critic_pretrain, max_steps))
        ray.get(refs)

    ray.get(ppo_trainer.fit.remote())
    ray.get(actor_model.async_save_model())

    if args.critic_pretrain and args.save_value_network:
        ray.get(critic_model.async_save_model())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Ray and vLLM
    parser.add_argument("--ref_num_nodes", type=int, default=1, help="number of nodes for reference")
    parser.add_argument("--ref_num_gpus_per_node", type=int, default=8, help="number of gpus per node for reference")
    parser.add_argument("--reward_num_nodes", type=int, default=1, help="number of nodes for reward model")
    parser.add_argument(
        "--reward_num_gpus_per_node", type=int, default=8, help="number of gpus per node for reward model"
    )
    parser.add_argument(
        "--colocate_actor_ref",
        action="store_true",
        default=False,
        help="whether to colocate reference and actor model, if true, they will share same gpus.",
    )

    parser.add_argument("--actor_num_nodes", type=int, default=1, help="number of nodes for actor")
    parser.add_argument("--actor_num_gpus_per_node", type=int, default=8, help="number of gpus per node for actor")
    parser.add_argument("--critic_num_nodes", type=int, default=1, help="number of nodes for critic")
    parser.add_argument("--critic_num_gpus_per_node", type=int, default=8, help="number of gpus per node for critic")
    parser.add_argument(
        "--colocate_critic_reward",
        action="store_true",
        default=False,
        help="whether to colocate critic and reward model, if true, they will share same gpus.",
    )
    parser.add_argument(
        "--colocate_all_models",
        action="store_true",
        default=False,
        help="whether to colocate all models (including vLLM engines), if true, they will share same gpus.",
    )

    # vLLM for text generation
    parser.add_argument(
        "--vllm_num_engines", type=int, default=None, help="number of vLLM Engines, set to 0 to disable vLLM"
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="tensor parallel size of vLLM Engine for multi-GPU inference",
    )
    parser.add_argument("--vllm_sync_backend", type=str, default="nccl", help="DeepSpeed -> vLLM weight sync backend")
    parser.add_argument("--vllm_sync_with_ray", action="store_true", default=False)
    parser.add_argument("--enable_prefix_caching", action="store_true", default=False)
    parser.add_argument("--enforce_eager", action="store_true", default=False, help="Disable CUDA graph in vLLM")
    parser.add_argument(
        "--vllm_enable_sleep",
        action="store_true",
        default=False,
        help="Enable sleep mode for vLLM when using --colocate_all_models",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.95,
        help="vLLM gpu_memory_utilization",
    )
    # Your Efficient RL Framework Secretly Brings You Off-Policy RL Training: https://fengyao.notion.site/off-policy-rl
    parser.add_argument("--enable_vllm_is_correction", action="store_true", default=False)
    parser.add_argument("--vllm_is_truncated_threshold", type=float, default=2)

    parser.add_argument("--async_train", action="store_true", default=False, help="Enable async training")

    # Checkpoints
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_ppo_ray")
    parser.add_argument("--save_hf_ckpt", action="store_true", default=False)
    parser.add_argument("--disable_ds_ckpt", action="store_true", default=False)
    parser.add_argument("--max_ckpt_num", type=int, default=1)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument(
        "--use_ds_universal_ckpt", action="store_true", help="Use deepspeed universal checkpoint", default=False
    )

    # DeepSpeed
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--deepcompile", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    ## Make EMA as an optional feature
    parser.add_argument("--enable_ema", action="store_true", help="Enable EMA checkpoint for the model.")
    parser.add_argument("--ema_beta", type=float, default=0.992, help="EMA beta coefficient")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--actor_init_on_gpu", action="store_true", default=False)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        help="Attention implementation (e.g., eager, flash_attention_2, flash_attention_3, kernels-community/vllm-flash-attn3)",
    )
    parser.add_argument("--use_liger_kernel", action="store_true", default=False, help="Enable Liger Kernel")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--overlap_comm", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument(
        "--deepspeed_enable_sleep",
        action="store_true",
        default=False,
        help="Enable sleep mode for deepspeed when using --colocate_all_models",
    )
    parser.add_argument("--ds_tensor_parallel_size", type=int, default=1, help="DeepSpeed tensor parallel size")

    # packing samples using Flash Attention2
    parser.add_argument("--packing_samples", action="store_true", default=False)

    # dynamic batch size
    parser.add_argument("--use_dynamic_batch", action="store_true", default=False)
    parser.add_argument("--rollout_max_tokens_per_gpu", type=int, default=None)
    parser.add_argument("--train_max_tokens_per_gpu", type=int, default=16192)

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    # Knowledge Tree parameters
    parser.add_argument("--storage_path", type=str, default="./storage")
    parser.add_argument("--specified_domain", type=str, default="Mathematics")
    parser.add_argument("--specified_domains_ratio", type=str, default=None, help="The ratio of specified domains")
    parser.add_argument("--model_name", type=str, default=None, help="model name for knowledge tree files")
    parser.add_argument("--web_corpus_path", type=str, default=None, help="Path to the web corpus JSON file")
    parser.add_argument("--max_levels", type=int, default=4, help="max levels of the knowledge tree")
    parser.add_argument("--only_sample", action="store_true", default=False, help="Only sample without expanding the tree")
    parser.add_argument("--tree_prompt_version", type=str, default="math", help="Tree prompt family: math, med, phy")
    parser.add_argument("--use_specified_backbone", action="store_true",
                help="whether to preload the hand-crafted Mathematics backbone")
    parser.add_argument("--select_top_k_nodes", type=int, default=1, help="Select top k nodes when exploring the knowledge tree")
    parser.add_argument("--select_min_count", type=int, default=2, help="Minimum repeated proposal count for accepting a new node")

    parser.add_argument("--multi_domain_version", type=str, default=None, help="Optional override for the domain tag used in output file names")
    parser.add_argument("--disable_thinking_mode_or_not", type=str, default=None, help="whether disable thinking mode")

    parser.add_argument("--explore_strategy", type=str, default="random", choices=["reward", "random", "random_reward"], help="Tree exploration strategy")
    parser.add_argument("--tree_window_size", type=str, default="global", help="Tree window size, either 'global' or an integer")

    parser.add_argument("--title_selection_mode", type=str, default="random", help="How to choose titles for sampled content")
    
    
    # define parameters for self-play
    parser.add_argument("--max_steps", type=int, default=50, help="max steps for self-play training")
    parser.add_argument(
        "--num_reasoner_samples", type=int, default=8, help="Number of reasoner samples used to score each challenger QA"
    )
    parser.add_argument(
        "--challenger_max_resample_attempts", type=int, default=16, help="max resample attempts for challenger"
    )
    parser.add_argument(
        "--challenger_n_candidates_per_prompt", type=int, default=32, help="multi-times for challenger generate the question"
    )
    
    parser.add_argument("--challenger_rm_mode", type=str, default="spice", help="Reward shaping mode for the challenger")
    parser.add_argument("--format_invalid_penalty", type=float, default=-0.1, help="Penalty for invalid challenger outputs")
    parser.add_argument("--clean_web", action="store_true", default=False, help="Whether to clean web corpus")
    parser.add_argument("--clean_web_stop_steps", type=int, default=10, help="Step after which web cleaning is enabled")
    parser.add_argument("--question_type_judge_start_step", type=int, default=-1, help="Step after which question-type judging is enabled")
    parser.add_argument("--cold_start_steps", type=int, default=10000, help="Cold-start threshold in training steps")
    parser.add_argument("--template_type", type=str, default=None, help="The template type for generation")
    parser.add_argument("--generate_qa_prompt", type=str, default="math", help="QA prompt family: math, med, phy")
    parser.add_argument("--instruction_len", type=int, default=2048, help="Instruction prefix length reserved inside the prompt budget")

    # PPO
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--rollout_batch_size", type=int, default=1024, help="Batch size for make experience")
    parser.add_argument(
        "--vllm_generate_batch_size", type=int, default=None, help="Batch size for vLLM generating samples"
    )
    parser.add_argument("--micro_rollout_batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--prompt_max_len", type=int, default=24576, help="Max tokens for each prompt")
    parser.add_argument("--generate_max_len", type=int, default=8192, help="Max tokens to generate in PPO")
    parser.add_argument("--max_len", type=int, default=None, help="deprecated max_len")
    parser.add_argument("--max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--max_data_tokens_num", type=int, default=5992, help="Max number of samples")
    
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--l2", type=float, default=0.0, help="weight decay loss")
    parser.add_argument("--ptx_coef", type=float, default=0.05, help="PPO-ptx loss coef")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--eps_clip_low_high", type=float, nargs=2, default=None, help="PPO-clip low and high")
    parser.add_argument("--dual_clip", type=float, default=None, help="Dual-clip PPO")
    parser.add_argument("--value_clip", type=float, default=0.5, help="PPO value clip range")
    parser.add_argument("--lambd", type=float, default=1, help="PPO GAE lambd")
    parser.add_argument("--gamma", type=float, default=1, help="PPO GAE gamma")
    parser.add_argument("--micro_train_batch_size", type=int, default=4, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normalization")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--full_determinism",
        action="store_true",
        default=False,
        help="Enable reproducible behavior during distributed training",
    )
    parser.add_argument("--freezing_actor_steps", type=int, default=-1, help="Used for critic initialization")
    parser.add_argument(
        "--n_samples_per_prompt", type=int, default=1, help="number of responses for each prompt in generation"
    )
    
    parser.add_argument("--save_value_network", action="store_true", default=False, help="Save critic model")
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6)
    parser.add_argument("--critic_learning_rate", type=float, default=9e-6)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_min_lr")
    parser.add_argument("--kl_target", type=float, default=None)
    parser.add_argument("--kl_horizon", type=int, default=10000)
    parser.add_argument("--init_kl_coef", type=float, default=0.01, help="KL penalty in PPO")
    parser.add_argument("--policy_loss_type", type=str, default="ppo", choices=["ppo", "gspo", "dr_grpo"])
    parser.add_argument(
        "--kl_estimator",
        type=str,
        default="k1",
        choices=["k1", "k2", "k3"],
        help=(
            "In GRPO, k3 is utilized as the loss function, while k2, when used as the loss, is nearly equivalent to k1."
        ),
    )
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument(
        "--entropy_loss_coef",
        type=float,
        default=None,
        help="Entropy loss coef, set to 0 means only enable entropy logs",
    )
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")
    parser.add_argument("--reward_clip_range", type=float, nargs=2, default=(-10, 10), help="Reward clip range")

    # Reinforce/GRPO, etc
    parser.add_argument(
        "--advantage_estimator",
        type=str,
        choices=["gae", "reinforce", "rloo", "reinforce_baseline", "group_norm", "dr_grpo"],
        default="gae",
        help="Choose advantage estimation method: gae, reinforce, rloo, reinforce_baseline, group_norm, dr_grpo",
    )
    parser.add_argument("--use_kl_loss", action="store_true", default=False, help="whether to use KL loss from GRPO")
    parser.add_argument(
        "--no_advantage_std_norm",
        action="store_true",
        default=False,
        help="disable dividing by std for advantages while keeping mean normalization",
    )
    parser.add_argument(
        "--overlong_buffer_len", type=float, default=None, help="reward with optional overlong penalty"
    )
    parser.add_argument("--overlong_penalty_factor", type=float, default=1, help="overlong penalty factor")

    # Context Parallel
    parser.add_argument("--ring_attn_size", type=int, default=1, help="Ring attention group size")
    parser.add_argument(
        "--ring_head_stride",
        type=int,
        default=1,
        help="the number of heads to do ring attention each time. "
        "It should be a divisor of the number of heads. "
        "A larger value may results in faster training but will consume more memory.",
    )

    #  Models
    parser.add_argument("--pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--remote_rm_url", type=str, default=None, help="remote RM API (HTTP)")
    parser.add_argument("--critic_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--value_head_prefix", type=str, default="score")
    parser.add_argument("--ref_reward_offload", action="store_true", default=False)
    parser.add_argument("--agent_func_path", type=str, default=None, help="Agent script path")

    # Custom dataset
    parser.add_argument("--prompt_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument(
        "--prompt_data_probs",
        type=str,
        default=None,
        help="sampling probs for datasets",
    )
    parser.add_argument("--prompt_split", type=str, default="train")
    parser.add_argument("--eval_dataset", type=str, default=None, help="Path to the evaluation dataset")
    parser.add_argument("--eval_split", type=str, default="train")
    parser.add_argument("--eval_temperature", type=float, default=0.6, help="Temperature for evaluation")
    parser.add_argument(
        "--eval_n_samples_per_prompt", type=int, default=4, help="Number of samples per prompt for evaluation"
    )
    
    parser.add_argument("--data_type", type=str, default="", help="dataset version")

    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--label_key", type=str, default=None, help="JSON dataset key")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_ppo")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="ppo_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # Dynamic filtering
    parser.add_argument("--dynamic_filtering", action="store_true", default=False, help="Enable dynamic filtering")
    parser.add_argument(
        "--dynamic_filtering_reward_range", nargs=2, default=(0, 1), type=float, help="Dynamic filtering rewards range"
    )

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    # performance tuning
    parser.add_argument("--perf", action="store_true", default=False)

    # ModelScope parameters
    parser.add_argument("--use_ms", action="store_true", default=False)

    args = parser.parse_args()

    args.specified_domain = args.specified_domain.split(",")
    args.specified_domains_ratio = None if args.specified_domains_ratio is None else [int(x) for x in args.specified_domains_ratio.split(",")]
    
    assert isinstance(args.specified_domain, list), "args.specified_domain should be a list"
    args.model_name = args.model_name if args.model_name else args.pretrain.split("/")[-1]
    
    
    if args.tree_window_size != "global":
        try:
            args.tree_window_size = int(args.tree_window_size)
        except:
            raise ValueError("tree_window_size should be 'global' or an integer")
    
    
    if args.use_specified_backbone:  # Use the specified tree backbone
        from knowledge_tree.Tree_backbone.specified_backbone import SPECIFIED_BACKBONE
    else:
        SPECIFIED_BACKBONE = None
    
    if "qwen3" in args.pretrain.lower():
        args.template_type = "qwen3"
    elif "octothinker" in args.pretrain.lower():
        args.template_type = "octothinker"
    else:
        raise ValueError("Currently only Qwen3 and OctoThinker models are supported.")

    # Validate arguments
    if args.eps_clip_low_high is None:
        args.eps_clip_low_high = (args.eps_clip, args.eps_clip)

    if args.agent_func_path:
        args.remote_rm_url = "agent"

    if args.advantage_estimator not in ["gae"]:
        args.critic_pretrain = None
    elif args.critic_pretrain is None:
        if not args.remote_rm_url:
            args.critic_pretrain = args.reward_pretrain.split(",")[0]
        else:
            args.critic_pretrain = args.pretrain

    if args.advantage_estimator in ["rloo", "reinforce_baseline", "group_norm"]:
        assert args.n_samples_per_prompt > 1, f"{args.advantage_estimator} requires n_samples_per_prompt > 1"

    if args.remote_rm_url:
        args.remote_rm_url = args.remote_rm_url.split(",")

    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.input_template and "\\n" in args.input_template:
        print(
            "[Warning] input_template contains \\n characters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    if args.ring_attn_size > 1:
        if not args.packing_samples:
            print("[Warning] --ring_attn_size > 1 requires --packing_samples.")
            args.packing_samples = True

    if args.use_dynamic_batch:
        if not args.packing_samples:
            print("[Warning] Please --packing_samples to accelerate when --use_dynamic_batch is enabled.")
            args.packing_samples = True
        if args.rollout_max_tokens_per_gpu is None:
            print("[Warning] Set --rollout_max_tokens_per_gpu to --train_max_tokens_per_gpu.")
            args.rollout_max_tokens_per_gpu = args.train_max_tokens_per_gpu

    if args.packing_samples:
        if "flash_attention" not in args.attn_implementation:
            print(
                "[Warning] Please use --attn_implementation with flash_attention to accelerate when --packing_samples is enabled."
            )
            args.attn_implementation = "flash_attention_2"
        assert args.vllm_num_engines > 0, "Only support `--packing_samples` with vLLM."

    if args.vllm_enable_sleep and not args.colocate_all_models:
        print("Set args.vllm_enable_sleep to False when args.colocate_all_models is disabled.")
        args.vllm_enable_sleep = False

    if args.colocate_all_models and args.async_train:
        print("[Warning] Using --colocate_all_models in async RLHF only colocates DeepSpeed models.")

    if args.async_train:
        assert not args.vllm_enable_sleep, "Async RLHF is not supported with --vllm_enable_sleep."

    if args.eval_dataset:
        assert args.remote_rm_url, "`--eval_dataset` is only supported with `--remote_rm_url`."

    if args.use_kl_loss:
        if args.kl_estimator not in ["k2", "k3"]:
            print(f"Recommend setting {args.kl_estimator} to 'k2' or 'k3' when using KL as a loss")
    else:
        if args.kl_estimator not in ["k1"]:
            print(f"Recommend setting {args.kl_estimator} to 'k1' when not using KL as a loss.")

    # Set vLLM generate_batch_size to rollout_batch_size if not specified
    if not args.vllm_generate_batch_size:
        args.vllm_generate_batch_size = args.rollout_batch_size

    if args.dynamic_filtering:
        assert (
            args.dynamic_filtering_reward_range[0] < args.dynamic_filtering_reward_range[1]
        ), "reward_clip_range[0] must be less than reward_clip_range[1]"
        # assert (
        #     args.remote_rm_url or args.agent_func_path
        # ), "remote_rm_url or agent_func_path must be specified when using dynamic filtering"
        assert (
            args.n_samples_per_prompt > 1
        ), "n_samples_per_prompt must be greater than 1 when using dynamic filtering"

    assert (
        args.n_samples_per_prompt * args.rollout_batch_size // args.micro_rollout_batch_size
        >= args.actor_num_nodes * args.actor_num_gpus_per_node // args.ring_attn_size // args.ds_tensor_parallel_size
    ), "The number of sample batches must be greater than or equal to the effective number of actor processes."

    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub

        # Patch hub to download models from modelscope to speed up.
        patch_hub()

    print(f"===================================== args =====================================")
    print(f"args:{args}")
    print(f"================================================================================")
    
    train(args)
