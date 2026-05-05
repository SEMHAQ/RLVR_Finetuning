"""Scalar feedback: format quality and step completeness scoring."""

import re


def scalar_score(completions, **kwargs):
    """Score completions on format and reasoning quality (0-1).

    Components:
    - Format (0-0.3): has ####, uses step markers
    - Step completeness (0-0.4): number of reasoning steps, has calculations
    - Expression clarity (0-0.3): math symbols, structured output

    Returns:
        List[float]: scores in [0, 1].
    """
    rewards = []
    for comp in completions:
        score = 0.0

        # Format score (0-0.3)
        if "####" in comp:
            score += 0.2
        if re.search(r"\n\d+[\.\)]\s", comp) or re.search(r"\n[-*]\s", comp):
            score += 0.1  # has numbered/bullet steps

        # Step completeness (0-0.4)
        lines = [l.strip() for l in comp.strip().split("\n") if l.strip()]
        num_lines = len(lines)
        if num_lines >= 3:
            score += 0.1
        if num_lines >= 5:
            score += 0.1

        # Has intermediate calculations (numbers and operators)
        calc_patterns = re.findall(r"\d+\s*[+\-*/=]\s*\d+", comp)
        if len(calc_patterns) >= 1:
            score += 0.1
        if len(calc_patterns) >= 3:
            score += 0.1

        # Expression clarity (0-0.3)
        # Uses math notation
        if re.search(r"\\[a-zA-Z]+", comp) or re.search(r"\$.*?\$", comp):
            score += 0.1
        # Has clear final answer section
        if "####" in comp:
            answer_part = comp.split("####")[-1].strip()
            if answer_part and len(answer_part) < 50:
                score += 0.1
        # Reasonable length (not too short, not too long)
        word_count = len(comp.split())
        if 20 <= word_count <= 300:
            score += 0.1

        rewards.append(min(score, 1.0))
    return rewards
