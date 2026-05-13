"""
SFT (Supervised Fine-Tuning) Baseline
======================================
用于与 GRPO 对比的监督微调 baseline。
使用 transformers.Trainer 兼容旧版 TRL。

Usage:
    python scripts/train_sft.py --model Qwen/Qwen2.5-Math-1.5B --use_lora
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model


def parse_args():
    parser = argparse.ArgumentParser(description="SFT baseline training")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--use_lora", action="store_true", default=False)
    parser.add_argument("--tag", type=str, default="sft_baseline")
    return parser.parse_args()


def main():
    args = parse_args()

    os.environ["WANDB_MODE"] = "offline"

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA
    if args.use_lora:
        print("Using LoRA")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    # Load dataset
    print("Loading GSM8K training data...")
    ds = load_dataset("openai/gsm8k", "main", split="train")

    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    # Tokenize
    system = "You are a math tutor. Solve the problem step by step, then put your final numerical answer after #### ."

    def tokenize(example):
        prompt = f"Question: {example['question']}\nSolution: "
        full_text = f"<|system|>\n{system}\n<|user|>\n{prompt}\n<|assistant|>\n{example['answer']}"
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    ds = ds.map(tokenize, remove_columns=ds.column_names)
    print(f"Training examples: {len(ds)}")

    # Output directory
    output_dir = f"outputs/{args.tag}"
    os.makedirs(output_dir, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        gradient_checkpointing=not args.use_lora,
        logging_steps=10,
        save_strategy="epoch",
        report_to="wandb",
        run_name=args.tag,
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
    )

    print(f"Starting SFT training: {args.tag}")
    trainer.train()

    # Save final model
    final_dir = f"{output_dir}/final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Model saved to: {final_dir}")


if __name__ == "__main__":
    main()
