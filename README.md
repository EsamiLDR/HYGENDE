# New_Hygende

HyGende is a lightweight project for hypothesis-driven LLM evaluation and inference on text classification datasets. It supports a three-stage workflow:
- Baseline evaluation using zero-shot or few-shot prompting.
- Hypothesis generation from training data.
- Inference using generated hypotheses and hypothesis selection.

## Repository Structure

- `run_baseline.py` - Run zero-shot / few-shot baseline evaluation.
- `run_generation.py` - Stage 1: generate candidate hypotheses using training samples.
- `run_inference.py` - Stage 2: select hypotheses and run inference on test data.
- `requirements.txt` - Python dependencies.
- `algorithm/` - Core algorithm modules.
  - `task.py` - dataset loading and prompt formatting.
  - `llm.py` - unified LLM adapter for API, vLLM, or HuggingFace backends.
  - `engine.py` - generation and inference pipeline logic.
  - `selector.py` - hypothesis selection strategies.
- `data/` - dataset directories with prompt file in YAML and data in JSON.

## Installation

1. Create a Python environment (recommended):

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Layout

Each dataset folder under `data/` contains:
- `config.yaml` - task name and prompt templates used by `TaskManager`.
- `train.json` - training samples.
- `test.json` - test samples.

The JSON format is expected to include:
- `content` - list of text samples.
- `label` - list of ground truth labels.
- optional `metadata` - list of metadata objects per sample.

## Usage

### 1. Baseline Evaluation

Run a standard zero-shot or few-shot baseline.

```bash
python run_baseline.py --dataset paddle --model_name gpt-4o --backend api --shots 0 --num_test 200
```

Common options:
- `--dataset` dataset name under `data/`.
- `--model_name` model identifier or path.
- `--backend` `api`, `vllm`, or `huggingface`.
- `--shots` number of few-shot examples.
- `--num_test` number of test samples to evaluate.

### 2. Hypothesis Generation

Generate candidate hypotheses using training data.

```bash
python run_generation.py --dataset harvard --model_name gpt-4o --backend api --num_train 200
```

Generated hypothesis checkpoints are saved to:

```
outputs/{dataset}/{model_tag}/hypotheses/train_{num_train}/
```

### 3. Inference

Run inference with a selected hypothesis repository.

```bash
python run_inference.py --dataset harvard --num_train 200 --model_name gpt-4o --backend api --strategy similarity --num_test 600 --k 10
```

If `--hypothesis_file` is not provided, the script auto-locates the latest hypothesis checkpoint in the output directory.

Inference results are saved to:

```
outputs/{dataset}/{model_tag}/result/train_{num_train}_test_{num_test}/inference_results.json
```

## Algorithm Overview

- `TaskManager` loads the dataset and formats prompts based on YAML templates.
- `UnifiedLLM` abstracts model backends and provides a unified `generate()` method.
- `HyGendeEngine` performs:
  - batched hypothesis generation,
  - hypothesis evaluation,
  - UCB-based repository management,
  - iterative sample processing,
  - checkpoint saving,
  - inference with selected hypotheses.
- `HypothesisSelector` supports strategies:
  - `similarity` - semantic similarity with sentence embeddings,
  - `weight` - choose hypotheses by reward/accuracy,
  - `random` - random selection.

## Notes

- The project expects prompt templates in `config.yaml` for generation, inference, and baseline prompts.
- Local model usage via `vllm` or `huggingface` requires corresponding models to be available.
- `OPENAI_API_KEY` and `OPENAI_API_BASE` environment variables are used when the `api` backend is selected.

## Example Workflow

1. Install dependencies.
2. Run generation on a dataset: `python run_generation.py --dataset harvard`.
3. Run inference: `python run_inference.py --dataset harvard --num_train 200`.
4. Optionally compare baseline with `python run_baseline.py --dataset harvard --shots 0`.
