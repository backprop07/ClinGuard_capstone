# ClinGuard Capstone Artifact

This repository is the cleaned companion artifact for the STAT4799 capstone report, *ClinGuard: An Evaluation Pipeline for Clinical Guardian Agents*. It contains the text-only MedQA-derived benchmark sample, selected evaluation outputs, and scripts used to reproduce the report figures.

## Contents

- `data/medqa_text/`: 300 MedQA-derived text-only benchmark cases and their raw source samples.
- `data/ontologies/`: warning and attack ontologies used by the benchmark.
- `configs/`: MedQA-only benchmark and model configuration snapshots.
- `results/`: selected result artifacts for GPT-5.4, DeepSeek-V3.2, and Qwen3.5-9B.
- `figures/`: report-ready figures generated from the copied artifacts.
- `scripts/`: validation and figure-generation scripts.
- `src/clinguard_capstone/`: minimal helper utilities for the capstone artifact.

## Reproducing Figures

From the repository root:

```bash
PYTHONPATH=src python scripts/validate_artifact.py
PYTHONPATH=src python scripts/make_figures.py
```

The scripts read only files in this repository. They do not depend on the future-publication `mcgb` workspace.

## Data Scope

The capstone artifact intentionally includes only the MedQA text-derived benchmark used for the undergraduate report. Multimodal cases, patient images, caches, virtual environments, API credentials, and unrelated experimental runs are excluded.

## Limitations

The benchmark cases are generated candidates derived from examination-style questions and have not been clinically verified by medical professionals. The artifact is therefore suitable for capstone evaluation, reproducibility, and methodological inspection, but not for clinical deployment.
