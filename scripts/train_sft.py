"""
SFT (Supervised Fine-Tuning) Baseline
======================================
用于与 GRPO 对比的监督微调 baseline。

Usage:
    python scripts/train_sft.py --model Qwen/Qwen2.5-Math-1.5B --use_lora
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer
from peft import LoraConfig


def parse_args():
    parser = argparse.ArgumentParser(description="SFT baseline training")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--use_lora", action="store_true", default=False)
    parser.add_argument("--tag", type=str, default="sft_baseline")
    return parser.parse_args()


def format_gsm8k(example):
    """Format GSM8K for SFT training."""
    system = "You are a math tutor. Solve the problem step by step, then put your final numerical answer after #### ."
    prompt = f"Question: {example['question']}\nSolution: "
    completion = example['answer']
    text = f"<|system|>\n{system}\n<|user|>\n{prompt}\n<|assistant|>\n{completion}"
    return {"text": text}


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

    # LoRA config
    peft_config = None
    if args.use_lora:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        )

    # Load dataset
    print("Loading GSM8K training data...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.map(format_gsm8k, remove_columns=ds.column_names)

    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    # Output directory
    output_dir = f"outputs/{args.tag}"
    os.makedirs(output_dir, exist_ok=True)

    # Training arguments
    from transformers import TrainingArguments
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
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        dataset_text_field="text",
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
