# num_generations 消融实验
# Usage: .\scripts\run_ablation_ngen.ps1

Write-Host "=== num_generations 消融实验 ==="
Write-Host "开始时间: $(Get-Date)"

foreach ($ngen in @(2, 4, 8)) {
    Write-Host ""
    Write-Host ">>> 训练 num_generations=$ngen"
    $bs = $ngen * 2
    python scripts/train_grpo.py `
        --model Qwen/Qwen2.5-Math-1.5B `
        --dataset gsm8k `
        --reward rule `
        --lr 2e-5 `
        --batch_size $bs `
        --num_generations $ngen `
        --max_completion_length 256 `
        --use_lora `
        --tag "ablation_ngen_$ngen"

    Write-Host ">>> 评测 num_generations=$ngen"
    python scripts/eval.py `
        --model "outputs/grpo_ablation_ngen_$ngen/final" `
        --dataset gsm8k `
        --tag "ablation_ngen_$ngen"
}

Write-Host ""
Write-Host "=== 实验完成 ==="
Write-Host "结束时间: $(Get-Date)"
