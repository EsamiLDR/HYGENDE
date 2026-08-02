import json
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class Sample:
    id: int
    content: str
    label: str
    meta_data: Optional[Dict[str, Any]] = field(default_factory=dict)

class TaskManager:
    """
    Dataset parsing and prompt template wrapper.
    """
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.task_name = self.config.get("task_name", "default_task")
        self.templates = self.config.get("prompt_templates", {})

    def load_dataset(self, data_path: str, max_samples: Optional[int] = None) -> List[Sample]:
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        contents = raw_data.get("content", [])
        labels = raw_data.get("label", [])
        
        # Handle optional metadata such as likes or retweets
        meta_list = raw_data.get("metadata", [{} for _ in range(len(contents))])

        samples = []
        limit = min(len(contents), max_samples) if max_samples else len(contents)
        
        for idx in range(limit):
            cnt = contents[idx]
            # If metadata exists, append it to the text representation
            meta = meta_list[idx] if idx < len(meta_list) else {}
            if meta:
                meta_str = " | ".join([f"{k}: {v}" for k, v in meta.items()])
                full_content = f"{cnt} [Meta Info: {meta_str}]"
            else:
                full_content = str(cnt)

            lbl = str(labels[idx])
            samples.append(Sample(id=idx, content=full_content, label=lbl, meta_data=meta))

        return samples

    def format_observation(self, sample: Sample) -> str:
        template = self.templates.get("observations", {}).get("multi_content", "${content}")
        return template.replace("${content}", sample.content).replace("${label}", sample.label)

    def format_batched_generation_prompt(self, samples: List[Sample], num_hypotheses: int = 5) -> Dict[str, str]:
        obs_list = [self.format_observation(s) for s in samples]
        obs_text = "\n".join(obs_list)

        gen_config = self.templates.get("batched_generation", {})
        sys_prompt = gen_config.get("system", "").replace("${num_hypotheses}", str(num_hypotheses))
        user_prompt = gen_config.get("user", "")\
            .replace("${observations}", obs_text)\
            .replace("${num_hypotheses}", str(num_hypotheses))

        return {"system": sys_prompt, "user": user_prompt}

    def format_inference_prompt(self, sample: Sample, hypothesis_text: str) -> Dict[str, str]:
        inf_config = self.templates.get("inference", {})
        sys_prompt = inf_config.get("system", "")
        
        # Retrieve metadata values
        likes_count = sample.meta_data.get("likes_count", "N/A")
        replies_count = sample.meta_data.get("replies_count", "N/A")
        retweets_count = sample.meta_data.get("retweets_count", "N/A")
        
        user_prompt = inf_config.get("user", "")\
            .replace("${hypothesis}", hypothesis_text)\
            .replace("${content}", sample.content)\
            .replace("${likes_count}", str(likes_count))\
            .replace("${replies_count}", str(replies_count))\
            .replace("${retweets_count}", str(retweets_count))
        
        return {"system": sys_prompt, "user": user_prompt}


    def format_baseline_prompt(self, sample: Sample, few_shot_samples: List[Sample] = None) -> Dict[str, str]:
        base_config = self.templates.get("few_shot_baseline", {})
        sys_prompt = base_config.get("system", "")
        
        prefix = ""
        if few_shot_samples:
            prefix_tpl = self.templates.get("few_shot_prefix", "")
            obs_str = "\n".join([self.format_observation(s) for s in few_shot_samples])
            prefix = f"{prefix_tpl}\n{obs_str}\n"

        user_prompt = base_config.get("user", "")\
            .replace("${few_shot_prefix}", prefix)\
            .replace("${observations}", "")\
            .replace("${content}", sample.content)

        return {"system": sys_prompt, "user": user_prompt}