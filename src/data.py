"""Data loading and preprocessing for GRPO training."""

import re
from datasets import load_dataset


SYSTEM_PROMPT = (
    "You are a math tutor. Solve the problem step by step, "
    "then put your final numerical answer after #### ."
)


def load_gsm8k(split="train"):
    """Load GSM8K dataset and format for GRPO training.

    Returns a dataset with 'prompt' column (list of chat messages).
    """
    ds = load_dataset("openai/gsm8k", "main", split=split)

    def format_prompt(example):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ]
        return {"prompt": messages, "answer": example["answer"]}

    ds = ds.map(format_prompt, remove_columns=ds.column_names)
    return ds


def extract_answer(text):
    """Extract the numerical answer after #### from GSM8K format.

    Returns the extracted string, or None if not found.
    """
    match = re.search(r"####\s*(.+)", text)
    if match:
        return match.group(1).strip()
    # Fallback: try to find the last number in the text
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]
    return None


def normalize_answer(answer_str):
    """Normalize answer string for comparison."""
    if answer_str is None:
        return None
    # Remove commas, spaces, dollar signs
    answer_str = answer_str.replace(",", "").replace("$", "").strip()
    # Try to convert to float
    try:
        return str(float(answer_str))
    except ValueError:
        return answer_str.lower()
