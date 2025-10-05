"""Complete Training and Evaluation Pipeline"""

import subprocess
import os
import argparse
from dotenv import load_dotenv

load_dotenv()


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\nERROR: {description} failed with code {result.returncode}")
        return False
    return True


def main(args):
    print("\n" + "="*80)
    print("RESPONSIBLE AI ALIGNMENT PIPELINE")
    print("="*80 + "\n")
    
    # Step 1: Prepare evaluation data
    if not args.skip_data_prep:
        if not run_command(
            "python src/data/prepare_data.py --num_samples 100 --output_file ./data/evaluation_set.jsonl",
            "STEP 1: Preparing Evaluation Dataset"
        ):
            return
    
    # Step 2: Baseline evaluation
    if not args.skip_baseline:
        if not run_command(
            "python src/evaluation/evaluate_model.py --config configs/training_config.yaml --num_samples 100",
            "STEP 2: Baseline Model Evaluation"
        ):
            return
    
    # Step 3: QLoRA training
    if not args.skip_qlora_train:
        if not run_command(
            "python src/training/train_qlora.py --config configs/training_config.yaml",
            "STEP 3: QLoRA Training"
        ):
            return
    
    # Step 4: QLoRA evaluation
    if not args.skip_qlora_eval:
        if not run_command(
            "python src/evaluation/evaluate_model.py --config configs/training_config.yaml --model_path ./models/llama3-qlora-adapter --num_samples 100",
            "STEP 4: QLoRA Model Evaluation"
        ):
            return
    
    # Step 5: DPO training (optional)
    if args.include_dpo and not args.skip_dpo_train:
        if not run_command(
            "python src/training/train_dpo.py --config configs/dpo_config.yaml",
            "STEP 5: DPO Training"
        ):
            return
    
    # Step 6: DPO evaluation (optional)
    if args.include_dpo and not args.skip_dpo_eval:
        if not run_command(
            "python src/evaluation/evaluate_model.py --config configs/dpo_config.yaml --model_path ./models/llama3-dpo-adapter --num_samples 100",
            "STEP 6: DPO Model Evaluation"
        ):
            return
    
    # Step 7: Compare results
    if not args.skip_comparison:
        compare_cmd = (
            "python src/evaluation/compare_results.py "
            "--baseline ./results/evaluation_baseline_100samples.csv "
            "--qlora ./results/evaluation_qlora_100samples.csv"
        )
        
        if args.include_dpo and os.path.exists("./results/evaluation_dpo_100samples.csv"):
            compare_cmd += " --dpo ./results/evaluation_dpo_100samples.csv"
        
        if not run_command(compare_cmd, "STEP 7: Comparing Results"):
            return
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\nResults saved in ./results/")
    print("Models saved in ./models/")
    print("\nNext steps:")
    print("1. Review comparison metrics in ./results/comparison_metrics.csv")
    print("2. Check visualizations in ./results/model_comparison.png")
    print("3. Analyze logs in W&B dashboard")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete alignment pipeline")
    
    # Pipeline control
    parser.add_argument("--include_dpo", action="store_true", help="Include DPO training and evaluation")
    
    # Skip options (for resuming)
    parser.add_argument("--skip_data_prep", action="store_true", help="Skip data preparation")
    parser.add_argument("--skip_baseline", action="store_true", help="Skip baseline evaluation")
    parser.add_argument("--skip_qlora_train", action="store_true", help="Skip QLoRA training")
    parser.add_argument("--skip_qlora_eval", action="store_true", help="Skip QLoRA evaluation")
    parser.add_argument("--skip_dpo_train", action="store_true", help="Skip DPO training")
    parser.add_argument("--skip_dpo_eval", action="store_true", help="Skip DPO evaluation")
    parser.add_argument("--skip_comparison", action="store_true", help="Skip comparison")
    
    args = parser.parse_args()
    main(args)

