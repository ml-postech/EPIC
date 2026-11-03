# Upstream acknowledgments

This repository developed from the RAISE research codebase. The retained generation/runtime scaffolding should be read with that provenance; EPIC adds the visual-program parsing, verification, action selection, and refinement pipeline. The original checkout referenced [RAISE: Requirement-Adaptive Evolutionary Refinement for Training-Free Text-to-Image Alignment](https://arxiv.org/abs/2603.00483).

External packages and model weights are installed or downloaded separately:

- [Diffusers](https://github.com/huggingface/diffusers), [Transformers](https://github.com/huggingface/transformers), and [LangGraph](https://github.com/langchain-ai/langgraph).
- [SAM3](https://github.com/facebookresearch/sam3) for object proposals and segmentation.
- [MiDaS](https://github.com/isl-org/MiDaS) for depth estimation.
- [Black Forest Labs](https://huggingface.co/black-forest-labs), [Qwen](https://huggingface.co/Qwen), [SigLIP2](https://huggingface.co/google/siglip2-base-patch16-224) for the supported image/VLM/verifier backbones.

These projects, checkpoints, and benchmark datasets retain their respective licenses and access terms. Installing EPIC does not grant model access. In particular, consult each model card when using a checkpoint for uses beyond the experiments. No third-party checkpoint is redistributed in this release.

Benchmark sources are listed in [benchmarks/README.md](benchmarks/README.md). Please cite the upstream methods and benchmarks used in your experiments.

## Redistributed code and benchmark inputs

`geneval_generate.py` is adapted from [GenEval](https://github.com/djghosh13/geneval), and `benchmarks/geneval_evaluation_metadata.jsonl` retains its prompt metadata. The original MIT license and copyright notice are included in [licenses/GenEval-MIT.txt](licenses/GenEval-MIT.txt). These notices remain applicable to the upstream material.

The root [LICENSE](LICENSE) covers the EPIC code released by MLLab @ Postech. Third-party code, model weights, and benchmark data retain their respective upstream terms; the root license does not replace those terms.
