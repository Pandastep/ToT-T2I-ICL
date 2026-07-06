# Tree-of-Thoughts Reasoning for Text-to-Image In-Context Learning

This repository contains the implementation of the paper:

**Tree-of-Thoughts Reasoning for Text-to-Image In-Context Learning**  
Stepanida Alekseeva, Jenifer Kalafatovich, and Seong-Whan Lee  
Accepted to **IEEE SMC 2026**.

## Overview

This project implements a Tree-of-Thoughts (ToT) reasoning framework for text-to-image in-context learning (T2I-ICL).

The main goal is to improve how multimodal large language models analyze in-context examples, infer the underlying transformation pattern, construct structured reasoning traces, and generate prompts for image synthesis.

The framework separates reasoning from image generation. A multimodal language model analyzes the in-context examples and produces a structured prompt, which is then passed to a frozen text-to-image generator.

The repository includes code for:

- Structured Baseline prompting
- Chain-of-Thought prompting
- Tree-of-Thoughts reasoning
- CoBSAT-compatible data loading
- Prompt construction
- Image generation
- CLIP and constraint satisfaction evaluation
- Branching-factor ablation
- Scoring-weight sensitivity analysis

## Method

The proposed ToT framework uses a modular four-stage reasoning pipeline:

1. **Scene analysis**  
   Identifies the shared visual context and recurring elements across the in-context examples.

2. **Attribute analysis**  
   Determines which aspect changes across examples and which elements remain stable.

3. **Stability checking**  
   Verifies that the generated hypothesis preserves invariant elements while applying the query-specific transformation.

4. **Final composition**  
   Constructs the final text prompt for image generation.

Compared with standard prompting and single-path Chain-of-Thought reasoning, the Tree-of-Thoughts formulation allows multiple candidate reasoning paths to be generated, scored, and selected before final prompt construction.

## Repository Structure

```text
ToT-T2I-ICL/
├── ablation/        Ablation-specific ToT scripts
├── analysis/        Analysis utilities
├── evaluation/      CLIP and CSR evaluation scripts
├── generation/      SEED runner and generation utilities
├── load_datasets/   CoBSAT prompt/data loading utilities
├── load_models/     SEED model loading wrapper
├── pipeline/        Pipeline dataclasses and orchestrator
├── reasoning/       Analyzer, hypothesis generation, and ToT reasoning modules
├── utils/           Auxiliary utilities
│
├── configs.py       CoBSAT-compatible task and prompt configuration
├── environment.py   Local path configuration
├── load_dataset.py  CoBSAT-compatible dataset loader
├── load_model_tot.py
├── image_generator.py
│
├── main3.py         Main Tree-of-Thoughts runner
├── main_bc.py       Structured Baseline and CoT runner
│
├── run_main3.py
├── run_baseline_main_bc.py
├── run_cot_main_bc.py
├── run_ablation.py
├── run_weight_ablation.py
├── run_eval_branch_ablation_clip_csr.py
├── run_eval_weight_ablation.py
│
├── requirements.txt
├── environment.yml
├── README.md
└── LICENSE
```

## CoBSAT-Compatible Data Loading

This repository keeps a lightweight CoBSAT-compatible data-loading layer, including:

```text
configs.py
load_dataset.py
load_datasets/
```

These files define the 10 CoBSAT task configurations, item spaces, and prompt index orders used to construct in-context examples and query samples.

The actual CoBSAT image dataset is **not included** in this repository. To run the experiments, place the dataset under:

```text
datasets/
```

The expected dataset structure follows the original CoBSAT organization.

## External Dependencies

This repository does **not** include:

- CoBSAT image data
- SEED model weights
- SEED/CoBSAT model configuration files
- Stable Diffusion weights
- Generated images
- Evaluation output files
- Human evaluation raw files

Users are responsible for obtaining external datasets and model checkpoints and complying with their respective licenses.

The expected local structure is:

```text
ToT-T2I-ICL/
├── datasets/
├── models/
│   └── SEED/
│       └── configs/
│           ├── tokenizer/
│           ├── transform/
│           └── llm/
```

The default SEED path is defined in `environment.py`:

```python
SEED_PROJECT_ROOT = os.environ.get(
    "SEED_PROJECT_ROOT",
    os.path.join(ROOT_DIR, "models", "SEED")
)
```

You can also set the path manually:

```bash
export SEED_PROJECT_ROOT=/path/to/models/SEED
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/ToT-T2I-ICL.git
cd ToT-T2I-ICL
```

Install dependencies with pip:

```bash
pip install -r requirements.txt
```

Alternatively, create a conda environment:

```bash
conda env create -f environment.yml
conda activate tot-t2i-icl
```

## Requirements

The code was developed for a CUDA GPU environment.

Core dependencies include:

- Python 3.10
- PyTorch
- Transformers
- Diffusers
- Accelerate
- PEFT
- Pillow
- NumPy
- Pandas
- Matplotlib
- tqdm
- CLIP-related dependencies

See `requirements.txt` and `environment.yml` for the complete environment specification.

## Running Experiments

### Tree-of-Thoughts

Run ToT on all 10 CoBSAT tasks:

```bash
python run_main3.py
```

This script calls `main3.py` and runs the Tree-of-Thoughts pipeline over the default CoBSAT evaluation setting.

### Structured Baseline

Run the structured baseline:

```bash
python run_baseline_main_bc.py
```

The structured baseline uses metadata-free pattern analysis and produces the same final structured prompt format as the CoT runner, but without explicit multi-step reasoning.

### Chain-of-Thought

Run the Chain-of-Thought baseline:

```bash
python run_cot_main_bc.py
```

The CoT runner uses a deterministic single-path CoT-style reasoning structure for comparison against ToT.

## Ablation Studies

### Branching-Factor Ablation

Run branching-factor ablation:

```bash
python run_ablation.py
```

This evaluates different branching factors:

```text
B = 1, 2, 3, 5
```

The default ablation setting uses selected CoBSAT tasks:

```text
tasks = 1, 3, 5, 7, 9
```

### Scoring-Weight Sensitivity Analysis

Run scoring-weight sensitivity analysis:

```bash
python run_weight_ablation.py
```

This evaluates the stability of the ToT scoring function under different scoring-weight configurations.

## Evaluation

The repository includes CLIP similarity and constraint satisfaction rate evaluation scripts.

### Evaluate Branching-Factor Ablation

```bash
python run_eval_branch_ablation_clip_csr.py
python collect_eval_branch_ablation_clip_csr.py
```

The first script runs the evaluation.  
The second script collects per-task results and aggregates them by branching factor.

### Evaluate Weight Sensitivity

```bash
python run_eval_weight_ablation.py
python collect_eval_weight_ablation.py
```

The first script runs CLIP/CSR evaluation for each scoring-weight setting.  
The second script aggregates the results by scoring mode.

## Output Structure

Experiment outputs are saved under result directories such as:

```text
results/
results_final/
results_ablation/
eval_branch_ablation_clip_csr/
eval_weight_ablation_clip_csr/
```

These folders are excluded from the public repository through `.gitignore`.

A typical sample output directory may contain:

```text
prompt.txt
reasoning_log.json
icl_used.txt
image.png
```

Depending on the script, generated images and logs may use slightly different filenames.

## Main Results

The main comparison in the paper evaluates Baseline, CoT, and ToT on CoBSAT.

| Method | CLIP Similarity | Constraint Satisfaction Rate |
|---|---:|---:|
| Baseline | 0.287 ± 0.032 | 0.508 ± 0.336 |
| CoT | 0.302 ± 0.031 | 0.547 ± 0.341 |
| ToT | **0.318 ± 0.030** | **0.775 ± 0.252** |

The Tree-of-Thoughts method achieves the best overall CLIP similarity and the highest constraint satisfaction rate.

## Human Evaluation

A human evaluation was conducted using three criteria:

- Example consistency
- Query alignment
- Joint quality

The ToT-based method was preferred over the alternatives in all three criteria:

| Criterion | ToT Preference Rate |
|---|---:|
| Example consistency | 59.5% |
| Query alignment | 68.1% |
| Joint quality | 65.5% |

The human evaluation used anonymized and randomized outputs from Baseline, CoT, and ToT.

Raw human evaluation files are not included in this repository.

## Branching-Factor Ablation

The branching-factor ablation evaluates the effect of increasing the number of candidate reasoning paths.

| Branching Factor | Internal Reasoning Score |
|---:|---:|
| B = 1 | 0.489 |
| B = 2 | 0.649 |
| B = 3 | 0.700 |
| B = 5 | 0.723 |

Increasing the branching factor improves the internal reasoning score, suggesting that broader candidate exploration helps the framework identify stronger reasoning paths.

## Scoring-Weight Sensitivity

The scoring-weight sensitivity analysis evaluates whether the ToT framework remains stable under different scoring configurations.

The results show that CLIP similarity and constraint satisfaction remain relatively stable across scoring settings, suggesting that the method is not overly dependent on a single scoring-weight configuration.

## Reproducibility Notes

This repository is intended to support research reproducibility at the code level.

However, exact reproduction requires access to:

- CoBSAT dataset files
- SEED model files
- Compatible GPU hardware
- External image-generation model weights
- The same or compatible dependency versions

Generated images may vary depending on hardware, CUDA version, model checkpoints, and random seed handling.

## Important Notes

This repository does not include large generated outputs, model weights, external datasets, or raw human evaluation files.

The following paths are intentionally excluded:

```text
datasets/
models/
results/
results_*/
eval_*/
.cache/
sd-cache/
__pycache__/
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{alekseeva2026tot,
  title={Tree-of-Thoughts Reasoning for Text-to-Image In-Context Learning},
  author={Alekseeva, Stepanida and Kalafatovich, Jenifer and Lee, Seong-Whan},
  booktitle={Proceedings of the IEEE International Conference on Systems, Man, and Cybernetics},
  year={2026}
}
```

## License

This repository is provided for academic research purposes. See `LICENSE` for details.

## Acknowledgements

This work builds on the CoBSAT benchmark setting and SEED-based multimodal generation infrastructure. External datasets, checkpoints, and model components remain subject to their original licenses and terms of use.#