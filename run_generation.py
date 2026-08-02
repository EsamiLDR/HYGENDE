import argparse
import logging
import os
from algorithm.task import TaskManager
from algorithm.llm import UnifiedLLM
from algorithm.engine import HyGendeEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    parser = argparse.ArgumentParser(description="HyGende Stage 1: Hypothesis Generation")
    
    # Add dataset name parameter as the main entry
    parser.add_argument("--dataset", type=str, default='harvard', help="Name of the dataset (e.g., paddle, fourdeptweet)")
    
    # Keep existing options optional and auto-fill from dataset
    parser.add_argument("--config_path", type=str, default=None, help="Path to config file (auto-filled if --dataset is set)")
    parser.add_argument("--train_path", type=str, default=None, help="Path to train data (auto-filled if --dataset is set)")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Model name or path for LLM (e.g., gpt-4o, deepseek-v4-flash)")
    parser.add_argument("--backend", type=str, default="api", choices=["auto", "api", "vllm", "huggingface"])
    parser.add_argument("--num_train", type=int, default=200, help="Number of training samples to use for hypothesis generation")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (auto-filled if --dataset is set)")

    args = parser.parse_args()

    # 1. Build default paths
    # Auto-fill config, train, and output paths from dataset if absent
    if args.config_path is None:
        args.config_path = f"./data/{args.dataset}/config.yaml"
    if args.train_path is None:
        args.train_path = f"./data/{args.dataset}/train.json"
    
    # Construct output directory structure: ./outputs/{dataset}/{model_name}/hypotheses/train_{num_train}/
    model_tag = args.model_name.split("/")[-1]
    if args.output_dir is None:
        args.output_dir = f"./outputs/{args.dataset}/{model_tag}/hypotheses/train_{args.num_train}/"

    # 2. Load task and training data
    task_mgr = TaskManager(args.config_path)
    train_samples = task_mgr.load_dataset(args.train_path, max_samples=args.num_train)

    # 3. Initialize LLM
    llm = UnifiedLLM(model_name_or_path=args.model_name, backend=args.backend)

    # 4. Run generation phase
    engine = HyGendeEngine(task_mgr, llm)
    
    logging.info(f"Output directory set to: {args.output_dir}")
    engine.run_generation_phase(
        train_samples=train_samples,
        output_dir=args.output_dir,
        model_tag=model_tag
    )

if __name__ == "__main__":
    main()
