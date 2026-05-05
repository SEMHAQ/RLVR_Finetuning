"""Reward functions for GRPO training.

This module provides different reward signal sources:
- rule_based: correctness verification (for math/code tasks)
- format: reward for following the expected output format
- scalar: numerical score from external evaluator
- ai_feedback: LLM-as-judge scoring

For Phase 0, only rule_based and format rewards are used.
Phase 2+ will add multi-source fusion.
"""

import re
from src.data import extract_answer, normalize_answer


def rule_based_reward(completions, answer=None, **kwargs):
    """Rule-based reward: check if the extracted answer matches ground truth.

    Args:
        completions: list of model completion strings
        answer: list of ground truth answer strings (from dataset)

    Returns:
        list of float rewards
    """
    rewards = []
    for completion, gt in zip(completions, answer):
        pred = extract_answer(completion)
        gt_norm = normalize_answer(extract_answer(gt))

        if pred is not None and gt_norm is not None:
            pred_norm = normalize_answer(pred)
            if pred_norm == gt_norm:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards


def format_reward(completions, **kwargs):
    """Format reward: bonus for including #### in the response.

    Args:
        completions: list of model completion strings

    Returns:
        list of float rewards (0.0 or 0.1)
    """
    rewards = []
    for completion in completions:
        if "####" in completion:
            rewards.append(0.1)
        else:
            rewards.append(0.0)
    return rewards


def combined_rule_reward(completions, answer=None, **kwargs):
    """Combined rule-based + format reward for GSM8K.

    Total reward = correctness (0 or 1) + format bonus (0 or 0.1)
    """
    correctness = rule_based_reward(completions, answer=answer)
    formatting = format_reward(completions)
    return [c + f for c, f in zip(correctness, formatting)]


# ============================================================
# Phase 2: Multi-source rewards (placeholder for now)
# ============================================================

def scalar_reward(completions, scores=None, **kwargs):
    """Scalar reward from an external scoring function.

    Args:
        completions: list of model completion strings
        scores: list of float scores (pre-computed)

    Returns:
        list of float rewards
    """
    if scores is None:
        return [0.0] * len(completions)
    return scores


def ai_feedback_reward(completions, prompt=None, **kwargs):
    """AI feedback reward using LLM-as-judge.

    This is a placeholder. In Phase 2, this will call an LLM API
    to score the completions.

    Args:
        completions: list of model completion strings
        prompt: the original prompt

    Returns:
        list of float rewards
    """
    # TODO: Phase 2 - implement LLM-as-judge scoring
    raise NotImplementedError("AI feedback reward not yet implemented")


def fused_reward(completions, answer=None, fusion_method="adaptive", **kwargs):
    """Fused multi-source reward (MSF-GRPO).

    Combines rule, AI, and scalar feedback with adaptive weighting.

    Args:
        completions: list of model completion strings
        answer: list of ground truth answer strings
        fusion_method: 'adaptive' (variance-aware) or 'fixed' (static weights)

    Returns:
        list of float rewards
    """
    from src.feedback.rule import rule_score
    from src.feedback.ai import ai_score
    from src.feedback.scalar import scalar_score
    from src.feedback.fusion import AdaptiveFusion

    sources = {
        "rule": (lambda c, **_: rule_score(c, answer=answer), 1.0),
        "ai": (ai_score, 0.8),
        "scalar": (scalar_score, 0.5),
    }

    fusion = AdaptiveFusion(sources, method=fusion_method)
    fused, _, _ = fusion.compute_rewards(completions, **kwargs)

    return fused
