"""Evaluation utilities for GRPO-trained models."""

import re
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data import extract_answer, normalize_answer, SYSTEM_PROMPT


def evaluate_gsm8k(model_path, split="test", max_new_tokens=512, batch_size=8):
    """Evaluate a model on GSM8K.

    Args:
        model_path: path to the trained model or HuggingFace model ID
        split: dataset split to evaluate on
        max_new_tokens: maximum tokens to generate
        batch_size: evaluation batch size

    Returns:
        dict with accuracy and per-example results
    """
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"  # must be left for decoder-only generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    ds = load_dataset("openai/gsm8k", "main", split=split)

    correct = 0
    total = 0
    results = []

    for i in tqdm(range(0, len(ds), batch_size), desc="Evaluating"):
        batch = ds[i : i + batch_size]

        prompts = []
        gt_answers = []
        for q, a in zip(batch["question"], batch["answer"]):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt_text)
            gt_answers.append(a)

        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
            )

        # Decode only the generated part
        generated = outputs[:, inputs["input_ids"].shape[1] :]
        completions = tokenizer.batch_decode(generated, skip_special_tokens=True)

        for completion, gt in zip(completions, gt_answers):
            pred = extract_answer(completion)
            gt_norm = normalize_answer(extract_answer(gt))
            pred_norm = normalize_answer(pred) if pred else None

            is_correct = pred_norm is not None and gt_norm is not None and pred_norm == gt_norm
            if is_correct:
                correct += 1
            total += 1

            results.append({
                "completion": completion,
                "predicted_answer": pred,
                "ground_truth": gt,
                "correct": is_correct,
            })

    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }
