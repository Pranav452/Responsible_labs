# Responsible AI Alignment Pipeline

QLoRA + DPO fine-tuning on Llama-3-8B with Anthropic HH-RLHF dataset.

**Metrics:** Toxicity (Detoxify BERT) + Helpfulness/Harmlessness (GPT-4o-mini Judge) + 3-Model Comparison Charts

---

## Quick Start (Lightning Studio A100)

### Step 1: Push to GitHub (on your Mac)
```bash
cd /Users/pranavnair/final_responisible
git init
git add .
git commit -m "alignment pipeline"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### Step 2: Clone in Lightning Studio
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install -r requirements.txt
```

### Step 3: Run Everything (3-4 hours)
```bash
python run_pipeline.py --include_dpo
```

---

## Step-by-Step Execution

```bash
# 1. Prepare data (1 min)
python src/data/prepare_data.py

# 2. Baseline eval (30-40 min)
python src/evaluation/evaluate_model.py --config configs/training_config.yaml --num_samples 100

# 3. Train QLoRA (20-30 min)
python src/training/train_qlora.py --config configs/training_config.yaml

# 4. Eval QLoRA (30-40 min)
python src/evaluation/evaluate_model.py --config configs/training_config.yaml --model_path ./models/llama3-qlora-adapter --num_samples 100

# 5. Train DPO (40-50 min)
python src/training/train_dpo.py --config configs/dpo_config.yaml

# 6. Eval DPO (30-40 min)
python src/evaluation/evaluate_model.py --config configs/dpo_config.yaml --model_path ./models/llama3-dpo-adapter --num_samples 100

# 7. Compare all (1 min)
python src/evaluation/compare_results.py \
    --baseline ./results/evaluation_baseline_100samples.csv \
    --qlora ./results/evaluation_qlora_100samples.csv \
    --dpo ./results/evaluation_dpo_100samples.csv
```

---

## Monitor

**GPU:** `watch -n 1 nvidia-smi`

**W&B:** https://wandb.ai/pranavnairop090-misogi/Responsible-AI-Alignment-A100-please-final

---

## Results

**Metrics:** `results/comparison_metrics.csv`

**Charts:** `results/model_comparison.png`

**Models:** `models/llama3-qlora-adapter/` and `models/llama3-dpo-adapter/`
# Responsible_labs
