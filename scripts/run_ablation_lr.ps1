# 学习率消融实验
# Usage: .\scripts\run_ablation_lr.ps1

Write-Host "=== 学习率消融实验 ==="
Write-Host "开始时间: $(Get-Date)"

foreach ($lr in @("1e-6", "5e-6", "1e-5", "2e-5")) {
    Write-Host ""
    Write-Host ">>> 训练 lr=$lr"
    python scripts/train_grpo.py `
        --model Qwen/Qwen2.5-Math-1.5B `
        --dataset gsm8k `
        --reward rule `
        --lr $lr `
        --batch_size 6 `
        --num_generations 3 `
        --max_completion_length 256 `
        --use_lora `
        --tag "ablation_lr_$lr"

    Write-Host ">>> 评测 lr=$lr"
    python scripts/eval.py `
        --model "outputs/grpo_ablation_lr_$lr/final" `
        --dataset gsm8k `
        --tag "ablation_lr_$lr"
}

Write-Host ""
Write-Host "=== 实验完成 ==="
Write-Host "结束时间: $(Get-Date)"
