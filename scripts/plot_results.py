"""
结果可视化脚本
==============
从 results/ 目录读取实验结果，生成对比图表。

Usage:
    python scripts/plot_results.py
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir="results"):
    """Load all eval results from results directory."""
    results = {}
    for fname in os.listdir(results_dir):
        if fname.startswith("eval_") and fname.endswith(".json"):
            with open(os.path.join(results_dir, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
                key = fname.replace("eval_", "").replace(".json", "")
                results[key] = data
    return results


def plot_lr_ablation(results, save_path="results/ablation_lr.png"):
    """Plot learning rate ablation results."""
    lr_keys = [k for k in results if k.startswith("ablation_lr_")]
    if not lr_keys:
        print("No lr ablation results found")
        return

    lrs = []
    accs = []
    for k in sorted(lr_keys):
        lr = k.replace("ablation_lr_", "").replace("_", ".")
        lrs.append(lr)
        accs.append(results[k]["accuracy"] * 100)

    plt.figure(figsize=(8, 5))
    plt.bar(lrs, accs, color="steelblue", alpha=0.8)
    plt.xlabel("Learning Rate")
    plt.ylabel("GSM8K Accuracy (%)")
    plt.title("Learning Rate Ablation (GRPO + LoRA)")
    plt.ylim(0, 100)
    for i, v in enumerate(accs):
        plt.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_data_ablation(results, save_path="results/ablation_data.png"):
    """Plot data size ablation results."""
    data_keys = [k for k in results if k.startswith("ablation_data_")]
    if not data_keys:
        print("No data ablation results found")
        return

    sizes = []
    accs = []
    for k in sorted(data_keys, key=lambda x: int(x.split("_")[-1])):
        size = int(k.split("_")[-1])
        sizes.append(str(size))
        accs.append(results[k]["accuracy"] * 100)

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, accs, "o-", color="steelblue", linewidth=2, markersize=8)
    plt.xlabel("Training Samples")
    plt.ylabel("GSM8K Accuracy (%)")
    plt.title("Data Size Ablation (GRPO + LoRA)")
    plt.ylim(0, 100)
    for i, v in enumerate(accs):
        plt.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_main_comparison(results, save_path="results/main_comparison.png"):
    """Plot main experiment comparison."""
    methods = []
    gsm8k_accs = []
    math500_accs = []

    # Baseline
    if "v3_lora_math500" in results or "grpo_v3" in results:
        # GRPO results
        pass

    # Collect all results
    key_mapping = {
        "baseline": "Baseline (No training)",
        "grpo_v3": "GRPO (Rule)",
        "grpo_msf": "MSF-GRPO",
        "sft_baseline": "SFT",
    }

    for key, name in key_mapping.items():
        matching = [k for k in results if k.startswith(key)]
        if matching:
            k = matching[0]
            methods.append(name)
            gsm8k_accs.append(results[k]["accuracy"] * 100)
            # Check for MATH-500
            math_key = k + "_math500"
            if math_key in results:
                math500_accs.append(results[math_key]["accuracy"] * 100)
            else:
                math500_accs.append(0)

    if not methods:
        print("No main results found")
        return

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, gsm8k_accs, width, label="GSM8K", color="steelblue", alpha=0.8)
    bars2 = ax.bar(x + width/2, math500_accs, width, label="MATH-500", color="coral", alpha=0.8)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Method Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.set_ylim(0, 100)

    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f"{height:.1f}%", ha="center", fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f"{height:.1f}%", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    results = load_results()
    print(f"Loaded {len(results)} results")

    plot_main_comparison(results)
    plot_lr_ablation(results)
    plot_data_ablation(results)

    print("\nAll results:")
    for k, v in sorted(results.items()):
        print(f"  {k}: {v['accuracy']*100:.2f}% ({v['correct']}/{v['total']})")


if __name__ == "__main__":
    main()
