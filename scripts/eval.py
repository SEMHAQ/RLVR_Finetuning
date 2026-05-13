"""
Evaluation Script
==================
Evaluate a trained model on GSM8K test set.

Usage:
    python scripts/eval.py --model outputs/grpo_baseline/final
    python scripts/eval.py --model Qwen/Qwen2.5-Math-1.5B  # eval base model
    python scripts/eval.py --model outputs/grpo_baseline/final --batch_size 4
    python scripts/eval.py --model outputs/grpo_baseline/final --tag grpo_v1
"""

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval import evaluate_gsm8k, evaluate_math500


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model on math datasets")
    parser.add_argument("--model", type=str, required=True,
                        help="Model path or HuggingFace model ID")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=["gsm8k", "math500"],
                        help="Dataset to evaluate on")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tag", type=str, default=None,
                        help="Tag for result filename (default: auto from model name)")
    parser.add_argument("--use_chat", action="store_true", default=False,
                        help="Force chat template prompting (default: auto-detect)")
    parser.add_argument("--no_chat", action="store_true", default=False,
                        help="Force plain text prompting (no chat template)")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Evaluating model: {args.model}")
    print(f"Dataset: {args.dataset}/{args.split}")

    # Determine use_chat: explicit flag > auto-detect
    if args.no_chat:
        use_chat = False
    elif args.use_chat:
        use_chat = True
    else:
        use_chat = None  # auto-detect by model name

    if args.dataset == "math500":
        results = evaluate_math500(
            model_path=args.model,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            use_chat=use_chat,
        )
    else:
        results = evaluate_gsm8k(
            model_path=args.model,
            split=args.split,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            use_chat=use_chat,
        )

    print(f"\n{'='*50}")
    print(f"Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    print(f"{'='*50}")

    # ---- Always save to results/ directory ----
    os.makedirs("results", exist_ok=True)

    # Generate filename from tag or model name
    dataset_prefix = f"{args.dataset}_" if args.dataset != "gsm8k" else ""
    if args.tag:
        fname = f"results/eval_{args.tag}.json"
    else:
        model_name = args.model.replace("/", "_").replace("\\", "_")
        model_short = model_name.split("_")[-1] if "_" in model_name else model_name
        fname = f"results/eval_{dataset_prefix}{model_short}.json"

    output_data = {
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "accuracy": round(results["accuracy"], 4),
        "correct": results["correct"],
        "total": results["total"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "examples": results["results"][:10],
    }

    with open(fname, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {fname}")


if __name__ == "__main__":
    main()
