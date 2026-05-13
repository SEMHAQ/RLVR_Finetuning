# 数据规模消融实验
# Usage: .\scripts\run_ablation_data.ps1

Write-Host "=== 数据规模消融实验 ==="
Write-Host "开始时间: $(Get-Date)"

foreach ($size in @(1000, 3000, 5000, 7500)) {
    Write-Host ""
    Write-Host ">>> 训练 data_size=$size"
    python scripts/train_grpo.py `
        --model Qwen/Qwen2.5-Math-1.5B `
        --reward rule `
        --lr 2e-5 `
        --batch_size 6 `
        --num_generations 3 `
        --max_completion_length 256 `
        --use_lora `
        --max_samples $size `
        --tag "ablation_data_$size"

    Write-Host ">>> 评测 data_size=$size"
    python scripts/eval.py `
        --model "outputs/grpo_ablation_data_$size/final" `
        --dataset gsm8k `
        --tag "ablation_data_$size"
}

Write-Host ""
Write-Host "=== 实验完成 ==="
Write-Host "结束时间: $(Get-Date)"
