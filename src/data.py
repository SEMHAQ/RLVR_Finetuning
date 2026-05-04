"""Data loading and preprocessing for GRPO training."""

import re
from datasets import load_dataset


SYSTEM_PROMPT = (
    "You are a math tutor. Solve the problem step by step, "
    "then put your final numerical answer after #### ."
)

# Plain text prompt for base models (no chat template)
BASE_PROMPT_TEMPLATE = "Question: {question}\nSolution: "


def load_gsm8k(split="train", use_chat_template=False):
    """Load GSM8K dataset and format for GRPO training.

    Args:
        split: dataset split
        use_chat_template: if True, use chat message format (for chat/instruct models).
                          if False, use plain text format (for base models).

    Returns a dataset with 'prompt' column.
    """
    ds = load_dataset("openai/gsm8k", "main", split=split)

    if use_chat_template:
        def format_chat(example):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
            ]
            return {"prompt": messages, "answer": example["answer"]}
        ds = ds.map(format_chat, remove_columns=ds.column_names)
    else:
        def format_base(example):
            prompt = BASE_PROMPT_TEMPLATE.format(question=example["question"])
            return {"prompt": prompt, "answer": example["answer"]}
        ds = ds.map(format_base, remove_columns=ds.column_names)

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
