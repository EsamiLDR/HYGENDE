import json
import argparse
import logging
import os
from algorithm.task import TaskManager
from algorithm.llm import UnifiedLLM
from algorithm.selector import HypothesisSelector
from algorithm.engine import HyGendeEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    parser = argparse.ArgumentParser(description="HyGende Stage 2: Inference")
    
    parser.add_argument("--dataset", type=str, default='harvard', help="Name of the dataset (e.g., paddle, fourdeptweet)")
    parser.add_argument("--num_train", type=int, default=200, help="Training sample count used in Stage 1, needed to locate hypothesis file")
    
    # Optional parameters; auto-derive if omitted
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--hypothesis_file", type=str, default=None, help="Path to hypothesis JSON (auto-located if not provided)")
    
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Model name or path for LLM (e.g., gpt-4o, deepseek-v4-flash)")
    parser.add_argument("--backend", type=str, default="api")
    parser.add_argument("--strategy", type=str, default="similarity", choices=["similarity", "weight", "random"])
    parser.add_argument("--num_test", type=int, default=600)
    parser.add_argument("--k", type=int, default=10)
    
    args = parser.parse_args()

    model_tag = args.model_name.split("/")[-1]

    # 1. Build default paths
    if args.config_path is None:
        args.config_path = f"./data/{args.dataset}/config.yaml"
    if args.test_path is None:
        args.test_path = f"./data/{args.dataset}/test.json"
    
    # Construct experiment directories
    exp_dir = f"./outputs/{args.dataset}/{model_tag}/hypotheses/train_{args.num_train}/"
    out_dir = f"./outputs/{args.dataset}/{model_tag}/result/train_{args.num_train}_test_{args.num_test}/"
    
    # Auto-locate hypothesis file
    if args.hypothesis_file is None:
        # Expected final checkpoint filename: hypotheses_{model_tag}_N{num_train}_step_{num_train}.json
        # Assumes training completed all num_train samples
        expected_hyp_file = os.path.join(exp_dir, f"hypotheses_{model_tag}_N{args.num_train}_step_{args.num_train}.json")
        
        if os.path.exists(expected_hyp_file):
            args.hypothesis_file = expected_hyp_file
            logging.info(f"Auto-located hypothesis file: {args.hypothesis_file}")
        else:
            # Fallback: use the latest JSON file in the directory
            if os.path.exists(exp_dir):
                files = [f for f in os.listdir(exp_dir) if f.startswith("hypotheses_") and f.endswith(".json")]
                if files:
                    # Sort filenames and use the last one (usually latest step)
                    files.sort()
                    args.hypothesis_file = os.path.join(exp_dir, files[-1])
                    logging.warning(f"Exact final checkpoint not found. Using latest available: {args.hypothesis_file}")
                else:
                    raise FileNotFoundError(f"No hypothesis files found in {exp_dir}. Please specify --hypothesis_file.")
            else:
                raise FileNotFoundError(f"Experiment directory {exp_dir} does not exist. Please run Stage 1 first.")

    # Output file saved in unified result directory
    output_file = os.path.join(out_dir, "inference_results.json")

    # 2. Load data and hypothesis repository
    task_mgr = TaskManager(args.config_path)
    test_samples = task_mgr.load_dataset(args.test_path, max_samples=args.num_test)
    
    with open(args.hypothesis_file, "r", encoding="utf-8") as f:
        hypotheses_repo = json.load(f)

    # 3. Initialize LLM and selector
    llm = UnifiedLLM(model_name_or_path=args.model_name, backend=args.backend)
    selector = HypothesisSelector(strategy=args.strategy)

    # 4. Run inference phase
    engine = HyGendeEngine(task_mgr, llm)
    engine.run_inference_phase(
        test_samples=test_samples,
        hypotheses_repo=hypotheses_repo,
        selector=selector,
        output_file=output_file,
        k=args.k
    )

if __name__ == "__main__":
    main()
