"""Rule-based feedback: answer correctness verification."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import extract_answer, normalize_answer


def rule_score(completions, answer=None, **kwargs):
    """Check if predicted answer matches ground truth.

    Returns:
        List[float]: 1.0 for correct, 0.0 for incorrect.
    """
    if answer is None:
        return [0.0] * len(completions)

    rewards = []
    for comp, gt in zip(completions, answer):
        pred = extract_answer(comp)
        gt_norm = normalize_answer(extract_answer(gt))
        if pred is not None and gt_norm is not None:
            pred_norm = normalize_answer(str(pred))
            rewards.append(1.0 if pred_norm == gt_norm else 0.0)
        else:
            rewards.append(0.0)
    return rewards


def format_score(completions, **kwargs):
    """Check if completion follows GSM8K format (contains ####).

    Returns:
        List[float]: 0.1 if formatted, 0.0 otherwise.
    """
    return [0.1 if "####" in comp else 0.0 for comp in completions]
