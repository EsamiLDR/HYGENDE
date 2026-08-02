import random
import numpy as np
from typing import List, Dict, Any
from algorithm.task import Sample

class HypothesisSelector:
    def __init__(self, strategy: str = "similarity", embedding_model_name: str = "./models/all-MiniLM-L6-v2"):
        self.strategy = strategy.lower()
        self.embedder = None
        
        if self.strategy == "similarity":
            try:
                from sentence_transformers import SentenceTransformer
                self.embedder = SentenceTransformer(embedding_model_name)
            except Exception as e:
                print(f"[Warning] SentenceTransformer load failed: {e}. Fallback to weight strategy.")
                self.strategy = "weight"

    def select(
        self, 
        hypotheses_dict: Dict[str, Dict[str, Any]], 
        test_sample: Sample, 
        k: int = 10
    ) -> List[str]:
        keys = list(hypotheses_dict.keys())
        if not keys:
            return []
        if len(keys) <= k:
            return [hypotheses_dict[k_]["hypothesis"] for k_ in keys]

        if self.strategy == "random":
            selected_keys = random.sample(keys, k)
            return [hypotheses_dict[k_]["hypothesis"] for k_ in selected_keys]

        elif self.strategy == "weight":
            sorted_keys = sorted(
                keys, 
                key=lambda x: hypotheses_dict[x].get("reward", hypotheses_dict[x].get("acc", 0.0)), 
                reverse=True
            )
            return [hypotheses_dict[k_]["hypothesis"] for k_ in sorted_keys[:k]]

        elif self.strategy == "similarity":
            # Disable progress bar during sample encoding
            sample_emb = self.embedder.encode(
                test_sample.content, 
                convert_to_tensor=True, 
                show_progress_bar=False
            )
            scores = []

            for key in keys:
                item = hypotheses_dict[key]
                correct_exs = item.get("correct_examples", [])
                
                # If no example history is available, use hypothesis similarity
                if not correct_exs:
                    # Disable progress bar during hypothesis encoding
                    hyp_emb = self.embedder.encode(
                        item["hypothesis"], 
                        convert_to_tensor=True, 
                        show_progress_bar=False
                    )
                    sim = float((sample_emb @ hyp_emb.T) / (sample_emb.norm() * hyp_emb.norm() + 1e-8))
                else:
                    ex_texts = []
                    for ex in correct_exs[:10]:  # limit to first 10 examples for speed
                        if isinstance(ex, dict) and "content" in ex:
                            ex_texts.append(ex["content"])
                        elif isinstance(ex, (list, tuple)) and len(ex) > 0:
                            ex_texts.append(str(ex[0]))

                    if not ex_texts:
                        hyp_emb = self.embedder.encode(
                            item["hypothesis"], 
                            convert_to_tensor=True, 
                            show_progress_bar=False
                        )
                        sim = float((sample_emb @ hyp_emb.T) / (sample_emb.norm() * hyp_emb.norm() + 1e-8))
                    else:
                        # Disable progress bar during example batch encoding
                        ex_embs = self.embedder.encode(
                            ex_texts, 
                            convert_to_tensor=True, 
                            show_progress_bar=False
                        )
                        sims = (sample_emb @ ex_embs.T) / (sample_emb.norm() * ex_embs.norm(dim=-1) + 1e-8)
                        sim = float(sims.mean())

                scores.append((key, sim))

            scores.sort(key=lambda x: x[1], reverse=True)
            selected_keys = [x[0] for x in scores[:k]]
            return [hypotheses_dict[k_]["hypothesis"] for k_ in selected_keys]

        else:
            raise ValueError(f"Unknown selection strategy: {self.strategy}")