"""Evaluation utilities for GRPO-trained models."""

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data import extract_answer, normalize_answer, SYSTEM_PROMPT, BASE_PROMPT_TEMPLATE, load_math500, extract_math_answer



def evaluate_gsm8k(model_path, split="test", max_new_tokens=256, batch_size=32, use_chat=None):
    """Evaluate a model on GSM8K.

    Args:
        model_path: path to the trained model or HuggingFace model ID
        split: dataset split to evaluate on
        max_new_tokens: maximum tokens to generate
        batch_size: evaluation batch size
        use_chat: force chat (True) or base (False) prompting.
                  None = auto-detect by model name (contains "Instruct"/"Chat" → chat, else base)

    Returns:
        dict with accuracy and per-example results
    """
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Determine prompting strategy
    if use_chat is None:
        name_lower = model_path.lower()
        use_chat = any(k in name_lower for k in ["instruct", "chat", "-it"])
    print(f"Prompting: {'chat/instruct' if use_chat else 'base (plain text)'}")

    ds = load_dataset("openai/gsm8k", "main", split=split)

    correct = 0
    total = 0
    results = []

    for i in tqdm(range(0, len(ds), batch_size), desc="Evaluating"):
        batch = ds[i : i + batch_size]

        prompts = []
        gt_answers = []
        for q, a in zip(batch["question"], batch["answer"]):
            if use_chat:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                ]
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt_text = BASE_PROMPT_TEMPLATE.format(question=q)
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


def evaluate_math500(model_path, max_new_tokens=512, batch_size=32, use_chat=None):
    """Evaluate a model on MATH-500.

    Args:
        model_path: path to the trained model or HuggingFace model ID
        max_new_tokens: maximum tokens to generate
        batch_size: evaluation batch size
        use_chat: force chat (True) or base (False) prompting.

    Returns:
        dict with accuracy and per-example results
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    if use_chat is None:
        name_lower = model_path.lower()
        use_chat = any(k in name_lower for k in ["instruct", "chat", "-it"])
    print(f"Prompting: {'chat/instruct' if use_chat else 'base (plain text)'}")

    ds = load_math500(split="test", use_chat_template=use_chat)

    correct = 0
    total = 0
    results = []

    for i in tqdm(range(0, len(ds), batch_size), desc="Evaluating MATH-500"):
        batch = ds[i : i + batch_size]

        prompts = []
        gt_answers = []
        for item in batch:
            p = item["prompt"]
            a = item["answer"]
            if use_chat and isinstance(p, list):
                prompt_text = tokenizer.apply_chat_template(
                    p, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt_text = p
            prompts.append(prompt_text)
            gt_answers.append(a)

        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
            )

        generated = outputs[:, inputs["input_ids"].shape[1] :]
        completions = tokenizer.batch_decode(generated, skip_special_tokens=True)

        for completion, gt in zip(completions, gt_answers):
            pred = extract_math_answer(completion)
            gt_norm = normalize_answer(extract_math_answer(gt))
            pred_norm = normalize_answer(pred) if pred else None

            is_correct = pred_norm is not None and gt_norm is not None and pred_norm == gt_norm
            if is_correct:
                correct += 1
            total += 1

            results.append({
                "completion": completion[:200],
                "predicted_answer": pred,
                "ground_truth": gt[:100],
                "correct": is_correct,
            })

    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }
