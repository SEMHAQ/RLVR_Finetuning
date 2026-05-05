"""AI feedback: rule-simulated LLM-as-judge scoring."""

import re


def ai_score(completions, **kwargs):
    """Simulate LLM-as-judge scoring for reasoning quality (0-1).

    Evaluates:
    - Step count appropriateness (3-8 steps optimal)
    - Has intermediate calculations
    - Logical coherence (no abrupt jumps)
    - Reasoning depth

    Returns:
        List[float]: scores in [0, 1].
    """
    rewards = []
    for comp in completions:
        score = 0.0

        # Step count analysis
        lines = [l.strip() for l in comp.strip().split("\n") if l.strip()]
        num_steps = len(lines)

        if 3 <= num_steps <= 8:
            score += 0.25  # optimal step count
        elif num_steps >= 2:
            score += 0.1   # acceptable
        # too few steps = shallow reasoning

        # Has intermediate calculations
        calc_patterns = re.findall(r"\d+\.?\d*\s*[+\-*/×÷=]\s*\d+\.?\d*", comp)
        if len(calc_patterns) >= 2:
            score += 0.25
        elif len(calc_patterns) >= 1:
            score += 0.1

        # Logical coherence: transitions between steps
        transitions = re.findall(
            r"(?:therefore|so|thus|hence|then|next|finally|first|second|third|because|since)",
            comp, re.IGNORECASE
        )
        if len(transitions) >= 2:
            score += 0.2
        elif len(transitions) >= 1:
            score += 0.1

        # Reasoning depth: mentions concepts or operations
        reasoning_words = re.findall(
            r"(?:total|sum|difference|product|quotient|remaining|each|per|average|multiply|divide|add|subtract)",
            comp, re.IGNORECASE
        )
        if len(reasoning_words) >= 2:
            score += 0.15
        elif len(reasoning_words) >= 1:
            score += 0.05

        # Has clear conclusion
        if "####" in comp:
            score += 0.15
        elif re.search(r"(?:answer|result|equals?|is)\s*[:=]?\s*\d", comp, re.IGNORECASE):
            score += 0.1

        rewards.append(min(score, 1.0))
    return rewards
