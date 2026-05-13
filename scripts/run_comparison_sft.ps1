# SFT vs GRPO 对比实验
# Usage: .\scripts\run_comparison_sft.ps1

Write-Host "=== SFT vs GRPO 对比实验 ==="
Write-Host "开始时间: $(Get-Date)"

# SFT baseline
Write-Host ""
Write-Host ">>> SFT 训练"
python scripts/train_sft.py `
    --model Qwen/Qwen2.5-Math-1.5B `
    --lr 2e-5 `
    --batch_size 6 `
    --epochs 3 `
    --use_lora `
    --tag "sft_baseline"

Write-Host ">>> SFT 评测 GSM8K"
python scripts/eval.py `
    --model outputs/sft_baseline/final `
    --dataset gsm8k `
    --tag "sft_baseline"

Write-Host ""
Write-Host ">>> SFT 评测 MATH-500"
python scripts/eval.py `
    --model outputs/sft_baseline/final `
    --dataset math500 `
    --tag "sft_baseline_math500"

Write-Host ""
Write-Host "=== 实验完成 ==="
Write-Host "结束时间: $(Get-Date)"
