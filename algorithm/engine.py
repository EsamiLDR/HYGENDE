import os
import re
import math
import json
import logging
from typing import List, Dict, Any
from algorithm.llm import UnifiedLLM
from algorithm.task import TaskManager, Sample
from algorithm.selector import HypothesisSelector

logger = logging.getLogger("HyGende.Engine")

class HyGendeEngine:
    def __init__(self, task_mgr: TaskManager, llm: UnifiedLLM, alpha: float = 0.5, gamma: float = 1.0):
        self.task_mgr = task_mgr
        self.llm = llm
        self.alpha = alpha  # UCB exploration balance factor
        self.gamma = gamma  # error buffer tolerance growth rate
        self.hypotheses_repo: Dict[str, Dict[str, Any]] = {}

    def _extract_label(self, raw_output: str) -> str:
        """Parse the predicted label from raw LLM output."""
        match = re.search(r"(?:Final answer|Answer):\s*(yes|no|[0-3])\b", raw_output, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        
        matches = re.findall(r"\b(yes|no|[0-3])\b", raw_output, re.IGNORECASE)
        if matches:
            return matches[-1].lower()
        return "unknown"

    def _parse_generated_hypotheses(self, raw_output: str) -> List[str]:
        # 1. Remove reasoning blocks like <think>...</think>
        raw_output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL)
        
        # 2. Strip Markdown code fences such as ```markdown or ```
        raw_output = re.sub(r"```[\w]*", "", raw_output)

        lines = raw_output.strip().split("\n")
        hyps = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove leading numbering, punctuation, and stray brackets
            cleaned = re.sub(r"^[\d\W]+\s*", "", line)
            cleaned = cleaned.strip("[]`\"' ")
            
            # Skip lines that are section headers or prompt artifacts
            if cleaned and len(cleaned) > 5 and not cleaned.lower().startswith("proposed hypotheses"):
                hyps.append(cleaned)
        
        if not hyps:
            # Log full raw output for debugging parsing issues
            logger.warning(f"⚠️ No valid hypotheses parsed from LLM output. Raw output full content:\n{raw_output}")
        else:
            logger.info(f"✅ Successfully parsed {len(hyps)} hypotheses.")
        
        return hyps


    def _update_ucb_scores(self, step_t: int):
        for hyp_key, data in self.hypotheses_repo.items():
            n_j = data["num_visits"]
            if n_j == 0:
                data["reward"] = data["acc"]
            else:
                ucb_bonus = self.alpha * math.sqrt(math.log(max(step_t, 1)) / n_j)
                data["reward"] = data["acc"] + ucb_bonus

    def _prune_hypotheses_repo(self, max_capacity: int):
        """Prune the hypothesis repository by UCB score, keeping the top candidates."""
        if len(self.hypotheses_repo) > max_capacity:
            sorted_keys = sorted(
                self.hypotheses_repo.keys(), 
                key=lambda k: self.hypotheses_repo[k]["reward"], 
                reverse=True
            )
            retained_keys = set(sorted_keys[:max_capacity])
            pruned_count = len(self.hypotheses_repo) - max_capacity
            
            self.hypotheses_repo = {k: self.hypotheses_repo[k] for k in retained_keys}
            logger.info(f"✂️ Hypothesis repository reached capacity limits. Pruned {pruned_count} lowest-reward hypotheses (Current Pool Size: {len(self.hypotheses_repo)}).")

    def _evaluate_hypothesis(self, hyp: str, eval_samples: List[Sample]) -> Dict[str, Any]:
        """Evaluate a hypothesis on a sample set and compute initial accuracy."""
        correct_examples = []
        for sample in eval_samples:
            inf_prompt = self.task_mgr.format_inference_prompt(sample, f"- {hyp}")
            out = self.llm.generate(inf_prompt["user"], inf_prompt["system"])
            pred_lbl = self._extract_label(out)
            if pred_lbl == sample.label:
                correct_examples.append({
                    "id": sample.id,
                    "label": sample.label,
                    "content": sample.content
                })
        
        acc = len(correct_examples) / len(eval_samples) if eval_samples else 0.0
        return {
            "hypothesis": hyp,
            "acc": acc,
            "num_visits": 1,
            "reward": acc,
            "correct_examples": correct_examples
        }

    def run_generation_phase(
        self, 
        train_samples: List[Sample], 
        output_dir: str, 
        model_tag: str,
        init_size: int = 10, 
        error_threshold_T: int = 10,
        k_select: int = 10,
        max_repo_size: int = 20  # Set the maximum hypothesis repository size
    ):
        os.makedirs(output_dir, exist_ok=True)
        total_train = len(train_samples)

        # 1. Initialize hypothesis set
        init_subset = train_samples[:init_size]
        prompt_dict = self.task_mgr.format_batched_generation_prompt(init_subset, num_hypotheses=5)
        raw_hyps_text = self.llm.generate(prompt_dict["user"], prompt_dict["system"], max_tokens=4096)
        init_hyps = self._parse_generated_hypotheses(raw_hyps_text)

        # Fail-safe: log and warn if initialization produced no hypotheses
        if not init_hyps:
            logger.error("❌ Initial hypothesis generation failed to produce any valid items! Please check prompt template or raw output format.")
            # Consider retrying or changing the prompt if needed

        for h in init_hyps:
            if h not in self.hypotheses_repo:
                self.hypotheses_repo[h] = self._evaluate_hypothesis(h, init_subset)
        
        self._update_ucb_scores(step_t=1)
        self._prune_hypotheses_repo(max_repo_size)  # enforce capacity limit

        error_buffer: List[Sample] = []
        logger.info(f"Starting Generation Phase with {total_train} samples (Max Repo Capacity: {max_repo_size})...")

        # 2. Incremental iterative learning
        for step, sample in enumerate(train_samples, start=1):
            keys = list(self.hypotheses_repo.keys())
            sorted_keys = sorted(keys, key=lambda x: self.hypotheses_repo[x]["reward"], reverse=True)
            top_k_keys = sorted_keys[:k_select]
            combined_hyp_text = "\n".join([f"- {self.hypotheses_repo[k]['hypothesis']}" for k in top_k_keys])

            inf_prompt = self.task_mgr.format_inference_prompt(sample, combined_hyp_text)
            out = self.llm.generate(inf_prompt["user"], inf_prompt["system"])
            pred_lbl = self._extract_label(out)

            is_correct = (pred_lbl == sample.label)

            if pred_lbl == "unknown":
                logger.warning(
                    f"[Gen {step}/{total_train}] Sample ID: {sample.id} - "
                    f"⚠️ Label parsing failed! Raw Output snippet: '{out[:100]}...'"
                )
            else:
                mark = "✓" if is_correct else "✗"
                logger.info(
                    f"[Gen {step}/{total_train}] Sample ID: {sample.id} | "
                    f"Pred: {pred_lbl} | GT: {sample.label} | Result: {mark}"
                )

            # Update Top-K hypothesis scores
            for k_key in top_k_keys:
                item = self.hypotheses_repo[k_key]
                item["num_visits"] += 1
                if is_correct:
                    item["correct_examples"].append({
                        "id": sample.id,
                        "label": sample.label,
                        "content": sample.content
                    })
                item["acc"] = len(item["correct_examples"]) / item["num_visits"]

            self._update_ucb_scores(step_t=step)

            # Manage dynamic threshold and error buffer
            actual_k = len(top_k_keys)
            dynamic_Nh = math.ceil(actual_k * (step / total_train) * self.gamma)
            dynamic_Nh = max(1, dynamic_Nh)

            if not is_correct:
                num_failed_hyps = actual_k
                
                if num_failed_hyps >= dynamic_Nh:
                    error_buffer.append(sample)
                
                # If error buffer reaches threshold, generate new hypotheses
                if len(error_buffer) >= error_threshold_T:
                    gen_prompt = self.task_mgr.format_batched_generation_prompt(error_buffer, num_hypotheses=5)
                    new_hyps_text = self.llm.generate(gen_prompt["user"], gen_prompt["system"])
                    new_hyps = self._parse_generated_hypotheses(new_hyps_text)

                    eval_samples_for_new = init_subset + error_buffer

                    for nh in new_hyps:
                        if nh not in self.hypotheses_repo:
                            self.hypotheses_repo[nh] = self._evaluate_hypothesis(nh, eval_samples_for_new)
                    
                    self._update_ucb_scores(step_t=step)
                    
                    # Prune repository if capacity is exceeded
                    self._prune_hypotheses_repo(max_repo_size)
                    
                    error_buffer.clear()

            # Save checkpoint periodically
            if step % 10 == 0 or step == total_train:
                ckpt_path = os.path.join(output_dir, f"hypotheses_{model_tag}_N{total_train}_step_{step}.json")
                with open(ckpt_path, "w", encoding="utf-8") as f:
                    json.dump(self.hypotheses_repo, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Saved hypothesis checkpoint to {ckpt_path}")
                logger.info(f"================ [Gen {step}/{total_train}] Current Hypotheses Pool (Total: {len(self.hypotheses_repo)}/{max_repo_size}) ================")
                for h_idx, hyp_text in enumerate(self.hypotheses_repo.keys(), start=1):
                    logger.info(f"  [{h_idx}] {hyp_text}")
                logger.info("=========================================================================================")

    def run_inference_phase(
        self, 
        test_samples: List[Sample], 
        hypotheses_repo: Dict[str, Dict[str, Any]], 
        selector: HypothesisSelector, 
        output_file: str,
        k: int = 10
    ) -> float:
        results = []
        correct_count = 0
        total_samples = len(test_samples)

        logger.info(f"Starting Inference Phase with {total_samples} samples...")

        for idx, sample in enumerate(test_samples, start=1):
            selected_hyps = selector.select(hypotheses_repo, sample, k=k)
            combined_hyp_text = "\n".join([f"- {h}" for h in selected_hyps])

            inf_prompt = self.task_mgr.format_inference_prompt(sample, combined_hyp_text)
            raw_output = self.llm.generate(inf_prompt["user"], inf_prompt["system"])
            pred_label = self._extract_label(raw_output)

            is_correct = (pred_label == sample.label)
            if is_correct:
                correct_count += 1

            if pred_label == "unknown":
                logger.warning(
                    f"[Inf {idx}/{total_samples}] Sample ID: {sample.id} - "
                    f"⚠️ Label parsing failed! Raw Output snippet: '{raw_output[:100]}...'"
                )
            else:
                mark = "✓" if is_correct else "✗"
                logger.info(
                    f"[Inf {idx}/{total_samples}] Sample ID: {sample.id} | "
                    f"Pred: {pred_label} | GT: {sample.label} | Result: {mark}"
                )

            results.append({
                "sample_id": sample.id,
                "content": sample.content,
                "ground_truth": sample.label,
                "selected_hypotheses": selected_hyps,
                "raw_llm_output": raw_output,
                "predicted_label": pred_label,
                "is_correct": is_correct
            })

        acc = correct_count / total_samples if total_samples else 0.0
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        summary = {
            "accuracy": acc,
            "total_samples": total_samples,
            "correct_samples": correct_count,
            "details": results
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"Inference Completed. Accuracy: {acc:.4f}. Saved results to {output_file}")
        return acc