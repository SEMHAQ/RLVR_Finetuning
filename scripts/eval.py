"""
Evaluation Script
==================
Evaluate a trained model on GSM8K test set.

Usage:
    python scripts/eval.py --model outputs/grpo_baseline/final
    python scripts/eval.py --model Qwen/Qwen2.5-Math-1.5B  # eval base model
    python scripts/eval.py --model outputs/grpo_baseline/final --batch_size 4
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval import evaluate_gsm8k


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model on GSM8K")
    parser.add_argument("--model", type=str, required=True,
                        help="Model path or HuggingFace model ID")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path (default: <model_path>/eval_results.json)")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Evaluating model: {args.model}")
    print(f"Dataset: gsm8k/{args.split}")

    results = evaluate_gsm8k(
        model_path=args.model,
        split=args.split,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    print(f"\n{'='*50}")
    print(f"Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    print(f"{'='*50}")

    # Save results
    if args.output is None:
        args.output = os.path.join(args.model, "eval_results.json")
        if not os.path.isdir(os.path.dirname(args.output)):
            args.output = "eval_results.json"

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "accuracy": results["accuracy"],
            "correct": results["correct"],
            "total": results["total"],
            "examples": results["results"][:10],  # save first 10 examples
        }, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
