# EPIC

[![Project Page](https://img.shields.io/badge/Project-Page-4285F4)](https://mtablo.github.io/EPIC_page/)
[![arXiv](https://img.shields.io/badge/arXiv-2605.11722-b31b1b)](https://arxiv.org/abs/2605.11722)

Official implementation of the NeurIPS 2027 paper **[EPIC: Efficient Predicate-Guided Inference-Time Control for Compositional Text-to-Image Generation](https://arxiv.org/abs/2605.11722)**.

EPIC is a training-free inference-time refinement framework for compositional text-to-image generation. It verifies generated images against a fixed visual program and uses the results to guide editing and resampling.

## 1. Install the image-generation environment

Run commands from the repository root on Linux x86_64 with Git, Conda, and an NVIDIA driver compatible with CUDA 12.1. The reference setup uses a 48 GB worker GPU for generation and verification, with separate GPUs for the VLM server.

```bash
conda env create -f epic_env.yml
conda activate epic_env
```

This installs EPIC with PyTorch 2.5.1/CUDA 12.1 and pinned dependencies from [pyproject.toml](pyproject.toml) and [requirements.txt](requirements.txt). The VLM server uses a separate environment below.

## 2. Models

| Role | Model |
| --- | --- |
| Image generation and editing | [FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) |
| Parsing, review, and semantic verification | [Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) |
| Object proposals and segmentation | [SAM3](https://huggingface.co/facebook/sam3) |
| Region/text matching | [SigLIP2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224) |
| Depth relations | [MiDaS DPT_Large](https://github.com/isl-org/MiDaS) |

Models are downloaded automatically on first use. For FLUX.2-klein-9B and SAM3, accept the Hugging Face access conditions and run `hf auth login` first.

## 3. Start the VLM server

In a separate terminal, create the VLM environment. Its vLLM wheel requires a CUDA 13.0-compatible driver; the image worker uses CUDA 12.1.

```bash
conda env create -f epic_vlm_env.yml
conda activate epic_vlm

CUDA_VISIBLE_DEVICES=0,1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_USE_FLASHINFER_SAMPLER=0 \
vllm serve Qwen/Qwen3-VL-32B-Instruct \
  --served-model-name Qwen/Qwen3-VL-32B-Instruct \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.92 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice \
  --max-model-len 32786
```

Adjust GPU IDs and tensor parallelism to your hardware. For a remote server, bind to an appropriate interface and pass its address through `--base_url`. Set `OPENAI_API_KEY` if the endpoint requires authentication.

Keep `--tool-call-parser qwen3_xml` and `--enable-auto-tool-choice` enabled for EPIC's structured responses.

## 4. Generate an image

In the image-generation terminal, use a GPU that is separate from the VLM server:

```bash
conda activate epic_env
export CUDA_VISIBLE_DEVICES=2
export MODEL=openai/Qwen/Qwen3-VL-32B-Instruct

python -m graph.graph_main \
  --base_url http://127.0.0.1:8000 \
  --prompts 'Two green apples to the left of two orange ceramic bowls.' \
  --llm_temperature 0 --max_round 32 \
  --saving_path outputs/example
```

To process a text file, replace `--prompts ...` with `--benchmark_txt_path path/to/prompts.txt` (one prompt per line).

Outputs include final images, JSON traces, and intermediate artifacts. If the round budget is exhausted without acceptance, EPIC saves the best partial candidate with `final_candidate_accepted=false`.

## 5. Run the benchmarks

With the environment, models, and VLM server ready, define these shared options:

```bash
conda activate epic_env
export CUDA_VISIBLE_DEVICES=2
EPIC_ARGS=(
  --base_url http://127.0.0.1:8000
  --llm_temperature 0
)
```

### GenEval and DrawBench

Prompt files are included. See [benchmark sources](benchmarks/README.md).

```bash
python geneval_generate.py "${EPIC_ARGS[@]}" \
  --metadata_file benchmarks/geneval_evaluation_metadata.jsonl \
  --n_samples 4 --outdir outputs/geneval/epic

python drawbench_generate.py "${EPIC_ARGS[@]}" \
  --benchmark_txt_path benchmarks/drawbench.txt \
  --n_samples 1 --outdir outputs/drawbench/epic
```

Images are saved as `<outdir>/<prompt_index>/samples/<sample_index>.png`. For GenEval scores, use the [official evaluator](https://github.com/djghosh13/geneval) in a separate environment.

### GenEval2

Download the official benchmark data:

```bash
git clone https://github.com/facebookresearch/GenEval2.git GenEval2
python geneval2_generate.py "${EPIC_ARGS[@]}" \
  --benchmark_data GenEval2/geneval2_data.jsonl \
  --max_round 32 \
  --n_samples 1 --outdir outputs/geneval2/epic
```

Pass `outputs/geneval2/epic/image_paths.json` to the [official Soft-TIFA evaluator](https://github.com/facebookresearch/GenEval2#evaluation).

### GenAI-Bench

Prepare the official prompt metadata, then generate:

```bash
python scripts/prepare_genai_bench.py
python genai_bench_generate.py "${EPIC_ARGS[@]}" \
  --benchmark_data benchmarks/genai_bench/genai_bench_1600.jsonl \
  --max_round 32 \
  --n_samples 1 --outdir outputs/genai_bench/epic
```

The preparation script downloads the [GenAI-Bench-1600 metadata](https://huggingface.co/datasets/BaiqiL/GenAI-Bench-1600). Evaluator inputs are written under `<outdir>/eval/`.

### T2I-CompBench++

```bash
git clone https://github.com/Karine-Huang/T2I-CompBench.git benchmarks/T2I-CompBench
python scripts/prepare_t2i_compbench.py
python t2i_compbench_generate.py "${EPIC_ARGS[@]}" \
  --benchmark_data benchmarks/t2i_compbench/t2i_compbenchpp_val.jsonl \
  --max_round 32 \
  --n_samples 1 --outdir outputs/t2i_compbench/epic
```

The `++` manifest includes all eight categories; use `t2i_compbench_val.jsonl` for the original six. Use `--n_samples 10` for the original ten-image protocol. Images are written to `<outdir>/eval/<category>/samples/`. Follow the [official evaluation instructions](https://github.com/Karine-Huang/T2I-CompBench#evaluation) in a separate environment.

### UniGenBench English-short

```bash
git clone https://github.com/CodeGoat24/UniGenBench.git UniGenBench
python unigenbench_generate.py "${EPIC_ARGS[@]}" \
  --benchmark_data UniGenBench/data/test_prompts_en.csv \
  --max_round 32 \
  --model_name EPIC --images_per_prompt 1 \
  --outdir outputs/unigenbench/epic \
  --eval_data_dir UniGenBench/eval_data
```

Images are written to `UniGenBench/eval_data/en/EPIC/`. Follow the [official UniGenBench evaluator](https://github.com/CodeGoat24/UniGenBench) for scoring.

Completed outputs are skipped when rerunning with the same settings and output directory. Use a new directory when changing settings. See each runner's `--help` for prompt ranges, sharding, and output controls.

## Options

Common options for direct generation and benchmark runners:

| Argument | Default | Effect |
| --- | --- | --- |
| `--seed` | `42` | Base image and VLM seed; runners derive per-sample seeds from it. |
| `--max_round` | `32` | Total rounds, including the initial generation. |
| `--llm_temperature` | `0.0` | Sampling for the parser, reviewer, rewriter, and in-loop auditor. |
| `--rewritten_prompt_count` | `8` | Number of requested prompt rewrites. |
| `--persistent_stuck_in_loop_auditor_streak` | `2` | Consecutive blocked rounds required before eligible predicates can be audited. |

For benchmark runners, `--n_samples` controls final images per prompt; each sample has its own round budget. UniGenBench uses `--images_per_prompt 1`.

See [HYPERPARAMETERS.md](HYPERPARAMETERS.md) for fixed model and verifier settings, and [THIRD_PARTY.md](THIRD_PARTY.md) for upstream acknowledgments.
