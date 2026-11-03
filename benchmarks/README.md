# Benchmark inputs

The release retains these small prompt inputs from the research checkout:

- `geneval_evaluation_metadata.jsonl`: [GenEval](https://github.com/djghosh13/geneval), prompt metadata for object/attribute/compositional evaluation.
- `drawbench.txt`: [DrawBench / Imagen](https://imagen.research.google/), one prompt per line.

Other benchmarks are obtained from their official sources using the commands in the root README:

- [GenEval2](https://github.com/facebookresearch/GenEval2)
- [GenAI-Bench-1600](https://huggingface.co/datasets/BaiqiL/GenAI-Bench-1600), converted with `scripts/prepare_genai_bench.py`
- [T2I-CompBench++](https://github.com/Karine-Huang/T2I-CompBench), converted with `scripts/prepare_t2i_compbench.py`
- [UniGenBench](https://github.com/CodeGoat24/UniGenBench), English-short CSV

Benchmark data and evaluator implementations remain subject to their upstream terms. Generated evaluation annotations, private analysis, and experiment outputs are not included.
