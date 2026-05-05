"""
GRPO Baseline Training Script
==============================
Phase 0: Reproduce GRPO on GSM8K with Qwen2.5-Math-1.5B

Usage:
    python scripts/train_grpo.py
    python scripts/train_grpo.py --config configs/grpo_baseline.yaml
    python scripts/train_grpo.py --model Qwen/Qwen2.5-Math-1.5B --num_epochs 1

GPU requirement: RTX 3090 24GB (with gradient checkpointing)
"""

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model

from src.data import load_gsm8k
from src.rewards import combined_rule_reward


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO Baseline Training")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--output_dir", type=str, default="outputs/grpo_baseline")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=9)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--num_generations", type=int, default=3,
                        help="Number of completions per prompt (G in GRPO)")
    parser.add_argument("--max_completion_length", type=int, default=384)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--beta", type=float, default=0.0,
                        help="KL penalty coefficient (0 = no KL)")
    parser.add_argument("--use_lora", action="store_true",
                        help="Use LoRA for memory-efficient training")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Limit training samples (0 = use all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--reward", type=str, default="combined_rule",
                        choices=["rule", "format", "combined_rule"],
                        help="Reward function to use")
    parser.add_argument("--wandb_project", type=str, default="unifb",
                        help="WandB project name (set to 'none' to disable)")
    return parser.parse_args()


def get_reward_fn(reward_type):
    """Get the reward function by name."""
    if reward_type == "rule":
        from src.rewards import rule_based_reward
        return rule_based_reward
    elif reward_type == "format":
        from src.rewards import format_reward
        return format_reward
    elif reward_type == "combined_rule":
        return combined_rule_reward
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")


def main():
    args = parse_args()

    # ---- WandB setup ----
    if args.wandb_project.lower() == "none":
        os.environ["WANDB_DISABLED"] = "true"
    else:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    # ---- Load model ----
    print(f"Loading model: {args.model}")
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }

    # LoRA: let trainer handle device placement; full: use device_map
    if not args.use_lora:
        model_kwargs["device_map"] = "auto"

    # Try flash_attention_2, fall back to default
    try:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    except Exception:
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Optional LoRA ----
    lora_config = None
    if args.use_lora:
        print(f"Applying LoRA (rank={args.lora_rank})")
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # ---- Load data ----
    # Auto-detect: model name contains "Instruct"/"Chat"/"-it" → use chat template
    name_lower = args.model.lower()
    use_chat = any(k in name_lower for k in ["instruct", "chat", "-it"])
    print(f"Prompting: {'chat/instruct' if use_chat else 'base (plain text)'}")

    print("Loading GSM8K training data...")
    train_dataset = load_gsm8k(split="train", use_chat_template=use_chat)
    if args.max_samples > 0:
        train_dataset = train_dataset.select(range(min(args.max_samples, len(train_dataset))))
    print(f"Training examples: {len(train_dataset)}")

    # ---- Reward function ----
    reward_fn = get_reward_fn(args.reward)
    print(f"Reward function: {args.reward}")

    # ---- Training config ----
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        num_generations=args.num_generations,
        beta=args.beta,
        loss_type="grpo",
        bf16=True,
        gradient_checkpointing=not args.use_lora,
        gradient_checkpointing_kwargs={"use_reentrant": False} if not args.use_lora else None,
        max_grad_norm=1.0,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        report_to="wandb" if args.wandb_project.lower() != "none" else "none",
        remove_unused_columns=False,
    )

    # ---- Trainer ----
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=reward_fn,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # ---- Train ----
    print("Starting GRPO training...")
    train_result = trainer.train()

    # ---- Save ----
    print(f"Saving model to {args.output_dir}/final")
    trainer.save_model(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")

    # ---- Save training summary to results/ ----
    os.makedirs("results", exist_ok=True)
    summary = {
        "model": args.model,
        "reward": args.reward,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum * args.num_generations,
        "num_generations": args.num_generations,
        "learning_rate": args.lr,
        "beta": args.beta,
        "use_lora": args.use_lora,
        "output_dir": args.output_dir,
        "total_steps": train_result.global_step if train_result else None,
        "train_loss": round(train_result.training_loss, 4) if train_result and hasattr(train_result, "training_loss") else None,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open("results/train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Training summary saved to: results/train_summary.json")
    print("Training complete!")


if __name__ == "__main__":
    main()
