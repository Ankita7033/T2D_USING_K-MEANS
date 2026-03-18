# Graph-Based Trajectory Modeling for Progression-Aware Type 2 Diabetes Subtyping

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MLHC 2026](https://img.shields.io/badge/Submitted-MLHC%202026-purple.svg)]()

Official implementation of **"Graph-Based Trajectory Modeling for Progression-Aware Type 2 Diabetes Subtyping"**

> **Submitted to:** Machine Learning for Healthcare (MLHC) 2026  
> **Status:** Under Review  
> **Data:** MIMIC-IV v2.2 (PhysioNet) + NHANES 2017–2020 (CDC)

---

## Overview

This repository provides the complete implementation of a trajectory-aware diabetes subtyping framework that identifies four clinically meaningful T2D subtypes from longitudinal EHR data. The framework integrates:

- **Multi-scale temporal encoding** — captures daily, weekly, and yearly disease dynamics using scale-specific LSTM encoders
- **Hybrid DTW-attention alignment** — handles irregularly sampled EHR observations
- **Graph-based phenotype-outcome fusion** — jointly reasons over patient trajectories, phenotypes, and clinical outcomes via Graph Attention Networks
- **Progression-aware clustering** — post-hoc K-Means with Markov transition modeling for disease trajectory forecasting
- **Cross-cohort external validation** — phenotype replication on NHANES 2017–2020 (N=1,263 outpatient T2D patients)

### Pipeline Architecture

```
Input: Longitudinal EHR (MIMIC-IV)
         │
Stage 1: Multi-Scale Temporal Encoding  (micro/meso/macro LSTM)
         │
Stage 2: Hybrid DTW-Attention Alignment (irregular sampling)
         │
Stage 3: Graph-Based Phenotype Fusion   (GAT + outcome nodes)
         │
Stage 4: Progression-Aware Clustering   (K-Means + Markov transitions)
         │
Output:  4 Trajectory-Aware Diabetes Subtypes
```

---

## Key Results

### Clustering Performance (Temporal Validation, N=8,127)

| Method | Silhouette ↑ | Davies-Bouldin ↓ | C-statistic |
|--------|-------------|------------------|-------------|
| **FCDT-TPFF (Ours)** | **0.41 ± 0.03** | **0.87 ± 0.08** | **0.78** |
| Static K-Means | 0.22 ± 0.02 | 1.42 ± 0.11 | 0.63 |
| Ahlqvist et al. | 0.28 ± 0.03 | 1.24 ± 0.09 | 0.68 |
| Single-Scale LSTM | 0.31 ± 0.04 | 1.14 ± 0.13 | 0.72 |
| Transformer | 0.29 ± 0.03 | 1.28 ± 0.10 | 0.71 |

### Identified Subtypes

| Cluster | Prevalence | Age | BMI | HbA1c | 2-Year Event Rate |
|---------|-----------|-----|-----|-------|-------------------|
| 1: Mild-Stable | 32% | 54y | 26 kg/m² | 6.8% | 8.2% |
| 2: Moderate-Unstable | 19% | 62y | 31 kg/m² | 8.1% | 14.7% |
| 3: Obese-Insulin Resistant | 28% | 56y | 38 kg/m² | 7.9% | 18.3% |
| 4: Advanced-High Risk | 21% | 68y | 29 kg/m² | 9.2% | 31.4% |

### Cross-Cohort Validation (NHANES 2017–2020)

All four subtypes replicated in an independent outpatient community cohort:

| Metric | MIMIC-IV | NHANES |
|--------|----------|--------|
| Silhouette score | 0.41 | 0.41 |
| Davies-Bouldin | 0.87 | 0.83 |
| Profile correlation ρ | — | 0.92 (p<0.001) |

---

## Repository Structure

```
FCDT-TPFF/
├── fcdt_run_pipeline.py        # Master pipeline orchestrator — start here
├── fcdt_tpff_data.py           # Stage 1: MIMIC-IV data extraction
├── fcdt_tpff_features.py       # Stage 2: Multi-scale feature engineering
├── fcdt_tpff_model.py          # Model architecture (LSTM + GAT)
├── fcdt_tpff_training.py       # Training, evaluation, all validations
├── fcdt_tpff_figures.py        # Figure generation (Figures 1–8)
├── graph_ablation.py           # Ablation study
├── cluster_stability.py        # Bootstrap cluster stability analysis
├── markov_validation.py        # Markov transition validation
├── temporal_validation.py      # Temporal split validation
├── missing_data_framework.py   # Missing data robustness analysis
├── statistical_rigor.py        # Statistical tests and CIs
├── reproducibility.py          # Seed management for reproducibility
├── nhanes_validation.py        # Cross-cohort NHANES replication
├── run_model.py                # Quick-run script (Windows-friendly)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── mimic-iv/                   # MIMIC-IV data (not included — see below)
│   ├── hosp/
│   └── icu/
│
├── nhanes_data/                # NHANES .xpt files (not included — see below)
│   ├── P_DEMO.xpt
│   ├── P_GHB.xpt
│   └── ...
│
├── processed_data/             # Auto-generated during pipeline run
│   ├── patient_features.pkl
│   ├── static_features.csv
│   ├── demographics.csv
│   ├── best_model.pt
│   ├── jbi_results.pkl
│   └── nhanes_validation_results.pkl
│
└── figures/                    # Auto-generated publication-ready figures
    ├── figure1_pipeline.png
    ├── figure2_silhouette_comparison.png
    ├── figure3_umap_clusters.png
    ├── figure4_kaplan_meier.png
    ├── figure5_treatment_response.png
    ├── figure6_ablation.png
    ├── figure7_missing_data.png
    └── figure8_nhanes_replication.png
```

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/Ankita7033/FCDT-TPFF.git
cd FCDT-TPFF
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Or using conda
conda create -n fcdt python=3.10
conda activate fcdt
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Data Access

### MIMIC-IV (Primary dataset)

MIMIC-IV is a restricted-access dataset requiring credentialing:

1. Visit [PhysioNet MIMIC-IV](https://physionet.org/content/mimiciv/2.2/)
2. Complete **CITI training** (Data or Specimens Only Research course)
3. Sign the **PhysioNet Credentialed Health Data License**
4. Download **MIMIC-IV v2.2** and extract to `./mimic-iv/`

Required files:
```
mimic-iv/
├── hosp/diagnoses_icd.csv.gz
├── hosp/patients.csv.gz
├── hosp/admissions.csv.gz
├── hosp/labevents.csv.gz
├── hosp/prescriptions.csv.gz
└── icu/chartevents.csv.gz
```

### NHANES 2017–2020 (External validation dataset)

NHANES is publicly available with no registration required. Download the following files from [CDC NHANES](https://wwwn.cdc.gov/nchs/nhanes/) and place in `./nhanes_data/`:

```bash
# Paste each URL in your browser to download
https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_DEMO.xpt
https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_GHB.xpt
https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_GLU.xpt
https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_BMX.xpt
https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_BPXO.xpt
https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_BIOPRO.xpt
https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_DIQ.xpt
https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_MCQ.xpt
https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_KIQ_U.xpt
```

---

## Quick Start

### Run Full MIMIC-IV Pipeline

```bash
python fcdt_run_pipeline.py --mimic_path ./mimic-iv/
```

This single command runs all 4 stages and generates all figures (~2–3 hours with GPU).

### Run NHANES External Validation

```bash
python nhanes_validation.py --nhanes_path ./nhanes_data/
```

Output: Section 4.9 results, Table 6, and Figure 8.

### Resume from Checkpoint

```bash
# Resume from feature engineering
python fcdt_run_pipeline.py --mimic_path ./mimic-iv/ --resume features

# Resume from training
python fcdt_run_pipeline.py --mimic_path ./mimic-iv/ --resume training

# Generate figures only
python fcdt_run_pipeline.py --figures_only
```

### Memory-Constrained Systems

```bash
# For systems with <32GB RAM
python fcdt_run_pipeline.py --mimic_path ./mimic-iv/ --batch_size 16 --num_epochs 25
```

---

## Reproducing Paper Results

All results in the paper are reproducible with `seed=42` (fixed in all scripts).

Expected outputs after full pipeline run:

| Metric | Expected Value |
|--------|---------------|
| Silhouette score | 0.41 ± 0.03 |
| Davies-Bouldin index | 0.87 ± 0.08 |
| C-statistic (outcomes) | 0.78 (95% CI: 0.75–0.81) |
| ARI stability | 0.89 ± 0.03 |
| NHANES profile correlation | ρ = 0.92 (p<0.001) |

If results differ significantly, verify:
- MIMIC-IV version is **v2.2** (not v1.0 or v2.0)
- Cohort size is **N = 8,127** after filtering
- Random seed is **42** (set automatically)

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 16 GB | 32 GB |
| Storage | 50 GB free | 100 GB free |
| GPU | — | NVIDIA, CUDA 11.0+ |
| Python | 3.8 | 3.10 |

| Configuration | Runtime |
|--------------|---------|
| GPU (A100/V100) | ~2–3 hours |
| CPU only | ~4–6 hours |

---

## Troubleshooting

**Out of memory:**
```bash
python fcdt_run_pipeline.py --mimic_path ./mimic-iv/ --batch_size 16
```

**CUDA not available:**
```bash
export CUDA_VISIBLE_DEVICES=""   # Linux/macOS
set CUDA_VISIBLE_DEVICES=        # Windows
```

**Missing MIMIC-IV files:**
Check that all 6 required `.csv.gz` files are present in `./mimic-iv/hosp/` and `./mimic-iv/icu/`.

**Import errors:**
```bash
pip install -r requirements.txt --upgrade
```

**Figures blank:**
```bash
export MPLBACKEND=Agg
python fcdt_tpff_figures.py
```

---

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@inproceedings{anonymous2026trajectory,
  title     = {Graph-Based Trajectory Modeling for Progression-Aware
               Type 2 Diabetes Subtyping},
  author    = {Anonymous},
  booktitle = {Proceedings of Machine Learning for Healthcare (MLHC)},
  year      = {2026},
  note      = {Under review}
}
```

*Full citation will be updated upon publication.*

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

**Data licenses:**
- MIMIC-IV: [PhysioNet Credentialed Health Data License](https://physionet.org/content/mimiciv/view-license/2.2/)
- NHANES: Public domain (CDC/US Government)

---

## Acknowledgments

- MIMIC-IV Team at MIT Laboratory for Computational Physiology
- PhysioNet for data hosting and infrastructure
- National Center for Health Statistics (CDC) for NHANES data

---

## Version History

### v2.0.0 — March 2026
- Added NHANES 2017–2020 cross-cohort external validation (Section 4.9)
- Added `nhanes_validation.py` — standalone replication script
- Updated repository structure and cleaned duplicate files
- Fixed placeholder citations with real published papers
- Added Figure 8: cross-cohort phenotype replication

### v1.0.0 — January 2026
- Initial release with MIMIC-IV pipeline
- 4-stage FCDT-TPFF architecture
- Complete ablation, stability, and missing data analyses

---

> **Note:** This repository is currently under anonymous review for MLHC 2026. The citation block above uses "Anonymous" as required by the double-blind review process. Full author information will be added upon publication.
