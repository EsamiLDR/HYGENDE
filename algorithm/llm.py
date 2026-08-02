import time
import os
import re
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger("HyGende.LLM")

class UnifiedLLM:
    """
    Unified LLM adapter layer.
    Supports API-based models and local models via vLLM or HuggingFace.
    """
    def __init__(
        self,
        model_name_or_path: str,
        backend: str = "auto",  # 'api', 'vllm', 'huggingface', 'auto'
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: float = 60.0,  # default request timeout
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85
    ):
        self.model_name_or_path = model_name_or_path
        self.backend = backend.lower()
        self.timeout = timeout
        self.client = None
        self.vllm_engine = None
        self.hf_model = None
        self.tokenizer = None

        self._init_backend(api_key, api_base, tensor_parallel_size, gpu_memory_utilization)

    def _init_backend(self, api_key: str, api_base: str, tp_size: int, gpu_util: float):
        if self.backend == "auto":
            if any(k in self.model_name_or_path.lower() for k in ["gpt-", "deepseek-", "claude-"]):
                self.backend = "api"
            else:
                self.backend = "vllm"

        if self.backend == "api":
            try:
                import openai
                key = api_key or os.getenv("OPENAI_API_KEY", "sk-7f6773db48014b8088161f17c6db23fa")
                base = api_base or os.getenv("OPENAI_API_BASE", "https://api.deepseek.com")
                self.client = openai.OpenAI(api_key=key, base_url=base, timeout=self.timeout)
                logger.info(f"Initialized API Client for model: {self.model_name_or_path} (timeout={self.timeout}s)")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize API client: {e}")

        elif self.backend == "vllm":
            try:
                from vllm import LLM, SamplingParams
                logger.info(f"Attempting to load model via vLLM: {self.model_name_or_path}")
                self.vllm_engine = LLM(
                    model=self.model_name_or_path,
                    tensor_parallel_size=tp_size,
                    gpu_memory_utilization=gpu_util,
                    trust_remote_code=True
                )
                self.SamplingParams = SamplingParams
                logger.info("vLLM initialized successfully.")
            except Exception as e:
                logger.warning(f"vLLM initialization failed ({e}). Falling back to HuggingFace Transformers...")
                self.backend = "huggingface"

        if self.backend == "huggingface":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            logger.info(f"Loading HuggingFace model and tokenizer: {self.model_name_or_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("HuggingFace model loaded successfully.")

    def generate(
        self, 
        prompt: str, 
        system_prompt: str = "", 
        max_tokens: int = 1024, 
        temperature: float = 0.01,
        max_retries: int = 5  # maximum retry attempts
    ) -> str:
        """Unified generation API with retry support."""
        if self.backend == "api":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Retry on timeout or network errors
            for attempt in range(1, max_retries + 1):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name_or_path,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        extra_body={"thinking": {"type": "disabled"}}
                    )
                    return response.choices[0].message.content.strip()
                except Exception as e:
                    logger.warning(f"⚠️ API Request failed or timed out (Attempt {attempt}/{max_retries}): {e}")
                    if attempt < max_retries:
                        sleep_time = 2 ** attempt  # exponential backoff (2s, 4s, 8s...)
                        logger.info(f"Waiting {sleep_time} seconds before retrying...")
                        time.sleep(sleep_time)
                    else:
                        logger.error("❌ Request failed after maximum retries.")
                        raise e

        elif self.backend == "vllm":
            full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n" if "qwen" in self.model_name_or_path.lower() else f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
            sampling_params = self.SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                stop=["<|im_end|>", "<|eot_id|>"]
            )
            outputs = self.vllm_engine.generate([full_prompt], sampling_params, show_progress_bar=False)
            return outputs[0].outputs[0].text.strip()

        elif self.backend == "huggingface":
            import torch
            full_text = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            inputs = self.tokenizer(full_text, return_tensors="pt").to(self.hf_model.device)
            with torch.no_grad():
                outputs = self.hf_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=max(temperature, 1e-2),
                    do_sample=(temperature > 0.01),
                    pad_token_id=self.tokenizer.pad_token_id
                )
            gen_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return gen_text.strip()

        raise ValueError(f"Unsupported backend: {self.backend}")