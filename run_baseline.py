import os
import json
import argparse
import logging
from algorithm.task import TaskManager
from algorithm.llm import UnifiedLLM
from algorithm.engine import HyGendeEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    parser = argparse.ArgumentParser(description="Baseline Evaluation: Zero-Shot / Few-Shot LLM")
    
    parser.add_argument("--dataset", type=str, default='paddle', help="Name of the dataset (e.g., paddle, fourdeptweet)")
    
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)
    
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--backend", type=str, default="api")
    parser.add_argument("--shots", type=int, default=0)
    parser.add_argument("--num_test", type=int, default=200)
    parser.add_argument("--output_file", type=str, default=None)
    
    args = parser.parse_args()

    model_tag = args.model_name.split("/")[-1]

    # 1. Build default input paths
    if args.config_path is None:
        args.config_path = f"./data/{args.dataset}/config.yaml"
    if args.train_path is None:
        args.train_path = f"./data/{args.dataset}/train.json"
    if args.test_path is None:
        args.test_path = f"./data/{args.dataset}/test.json"

    # 2. Create output directory and default result file
    baseline_dir = f"./outputs/{args.dataset}/{model_tag}/baseline/"
    
    if args.output_file is None:
        os.makedirs(baseline_dir, exist_ok=True)
        filename = f"{model_tag}_test_{args.num_test}_{args.shots}_shot_result.json"
        args.output_file = os.path.join(baseline_dir, filename)
    else:
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    # 3. Load data
    task_mgr = TaskManager(args.config_path)
    test_samples = task_mgr.load_dataset(args.test_path, max_samples=args.num_test)
    few_shot_samples = []
    if args.shots > 0:
        few_shot_samples = task_mgr.load_dataset(args.train_path, max_samples=args.shots)

    # 4. Initialize model and engine
    llm = UnifiedLLM(model_name_or_path=args.model_name, backend=args.backend)
    engine = HyGendeEngine(task_mgr, llm)

    # 5. Evaluate samples and log results
    results = []
    correct_count = 0
    total_samples = len(test_samples)
    logging.info(f"Starting Baseline ({args.shots}-shot) evaluation on {args.dataset} for {total_samples} samples...")
    
    for idx, sample in enumerate(test_samples, start=1):
        prompt_dict = task_mgr.format_baseline_prompt(sample, few_shot_samples=few_shot_samples)
        raw_output = llm.generate(prompt_dict["user"], prompt_dict["system"])
        pred_label = engine._extract_label(raw_output)
        is_correct = (pred_label == sample.label)
        if is_correct:
            correct_count += 1

        if pred_label == "unknown":
            logging.warning(
                f"[Base {idx}/{total_samples}] Sample ID: {sample.id} - "
                f"⚠️ Label parsing failed! Raw Output snippet: '{raw_output[:100]}...'"
            )
        else:
            mark = "✓" if is_correct else "✗"
            logging.info(
                f"[Base {idx}/{total_samples}] Sample ID: {sample.id} | "
                f"Pred: {pred_label} | GT: {sample.label} | Result: {mark}"
            )
            
        results.append({
            "sample_id": sample.id,
            "content": sample.content,
            "ground_truth": sample.label,
            "raw_llm_output": raw_output,
            "predicted_label": pred_label,
            "is_correct": is_correct
        })

    acc = correct_count / total_samples if total_samples else 0.0

    # 6. Save results
    summary = {
        "model_name": args.model_name,
        "num_test": total_samples,
        "shots": args.shots,
        "accuracy": acc,
        "correct_samples": correct_count,
        "details": results
    }
    
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    logging.info(f"Baseline ({args.shots}-shot) Completed. Accuracy: {acc:.4f}. Saved to {args.output_file}")

if __name__ == "__main__":
    main()
