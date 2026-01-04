# Graph-Based Trajectory Modeling for Type 2 Diabetes Subtyping

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **"Graph-Based Trajectory Modeling for Progression-Aware Type 2 Diabetes Subtyping"**

> **Authors:** Ankita Maji, Shashwat Vijayvergiya, Vikas Kumar Yadav  
> **Institution:** Lovely Professional University  
> **Paper:** [Link to paper]

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Usage Guide](#usage-guide)
- [Reproducing Paper Results](#reproducing-paper-results)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

---

## 🔬 Overview

This repository provides a complete implementation of our trajectory-aware diabetes subtyping framework integrating:

- **Multi-scale temporal encoding** (daily, weekly, yearly dynamics)
- **Hybrid DTW-attention alignment** for irregular EHR data
- **Graph-based phenotype-outcome fusion** 
- **Progression-aware clustering** with Markov transitions

The framework identifies four clinically meaningful diabetes subtypes with differential treatment responses and complication risks from longitudinal electronic health record data.

### Architecture Diagram

```
Stage 1: Multi-Scale Temporal Encoding
    ↓
Stage 2: DTW-Attention Alignment
    ↓
Stage 3: Graph Neural Network Fusion
    ↓
Stage 4: Progression-Aware Clustering
```

---

## 🎯 Key Results

### Clustering Performance (Temporal Validation)

| Method | Silhouette ↑ | Davies-Bouldin ↓ | C-statistic |
|--------|-------------|------------------|-------------|
| **FCDT-TPFF** | **0.41±0.03** | **0.87±0.08** | **0.78** |
| Static K-Means | 0.22±0.02 | 1.42±0.11 | 0.63 |
| Ahlqvist | 0.28±0.03 | 1.24±0.09 | 0.68 |
| LSTM | 0.31±0.04 | 1.14±0.13 | 0.72 |

### Identified Subtypes

| Cluster | Prevalence | Age | BMI | HbA1c | 2-Year Event Rate |
|---------|-----------|-----|-----|-------|-------------------|
| 1: Mild-Stable | 32% | 54y | 26 | 6.8% | 8.2% |
| 2: Moderate-Unstable | 19% | 62y | 31 | 8.1% | 14.7% |
| 3: Obese-Insulin Resistant | 28% | 56y | 38 | 7.9% | 18.3% |
| 4: Advanced-High Risk | 21% | 68y | 29 | 9.2% | 31.4% |

### Treatment Response Associations

**Cluster 3 (Obese-IR):** 1.4 percentage point greater HbA1c reduction with SGLT2 inhibitors vs. metformin (p<0.001) in observational analyses.

---

## 💻 System Requirements

### Hardware

- **RAM:** 32GB minimum recommended
- **Storage:** 100GB+ free space (for MIMIC-IV dataset)
- **GPU:** Optional (NVIDIA with CUDA 11.0+)
- **CPU:** Multi-core processor recommended

### Software

- **Python:** 3.8 - 3.10
- **Operating System:** Linux, macOS, or Windows
- **CUDA:** 11.0+ (optional, for GPU acceleration)

### Estimated Runtime

| Configuration | Total Time |
|--------------|------------|
| **32GB RAM + GPU** | ~2-3 hours |
| **32GB RAM + CPU only** | ~4-6 hours |

---

## 🚀 Installation

### 1. Clone Repository

```bash
git clone https://github.com/[your-username]/diabetes-subtyping.git
cd diabetes-subtyping
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n fcdt python=3.10
conda activate fcdt
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### Requirements File (`requirements.txt`)

```txt
# Core dependencies
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0

# Machine learning
torch>=2.0.0
scikit-learn>=1.0.0
umap-learn>=0.5.3

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0

# Survival analysis
lifelines>=0.27.0

# Utilities
tqdm>=4.62.0
psutil>=5.8.0

# Optional: GPU support
# torch-cuda>=11.8
```

---

## ⚡ Quick Start

### Step 1: Obtain MIMIC-IV Data

1. Visit [PhysioNet MIMIC-IV](https://physionet.org/content/mimiciv/2.2/)
2. Complete required **CITI training** course
3. Sign **data use agreement**
4. Download **MIMIC-IV v2.2** dataset
5. Extract to a directory (e.g., `/data/mimic-iv-2.2/`)

### Step 2: Run Complete Pipeline

```bash
python fcdt_master_script.py --mimic_path /data/mimic-iv-2.2/
```

This single command will:
1. ✅ Extract T2DM cohort (8,127 patients)
2. ✅ Engineer multi-scale features
3. ✅ Train FCDT-TPFF model
4. ✅ Evaluate against baselines
5. ✅ Generate all paper figures

### Step 3: View Results

```bash
# Figures saved to ./figures/
ls figures/

# Results saved to ./processed_data/
ls processed_data/evaluation_results.csv
```

---

## 📁 Repository Structure

```
diabetes-subtyping/
├── fcdt_master_script.py          # 🎯 Main pipeline orchestrator
├── fcdt_tpff_data.py              # 📊 Stage 1: Data extraction
├── fcdt_tpff_features.py          # 🔧 Stage 2: Feature engineering
├── fcdt_tpff_model.py             # 🧠 Stage 3: Model architecture
├── fcdt_tpff_training.py          # 🎓 Stage 4: Training & evaluation
├── fcdt_tpff_figures.py           # 📈 Stage 5: Figure generation
├── requirements.txt               # 📦 Python dependencies
├── README.md                      # 📖 This file
│
├── processed_data/                # 💾 Generated data (auto-created)
│   ├── cohort.csv
│   ├── labs.csv
│   ├── demographics.csv
│   ├── patient_features.pkl
│   └── evaluation_results.csv
│
├── figures/                       # 📊 Generated figures (auto-created)
│   ├── figure1_pipeline.png
│   ├── figure2_silhouette_comparison.png
│   ├── figure3_umap_clusters.png
│   ├── figure4_kaplan_meier.png
│   ├── figure5_treatment_response.png
│   ├── figure6_ablation.png
│   └── figure7_missing_data.png
│
└── models/                        # 🤖 Saved models (auto-created)
    └── fcdt_tpff_best.pt
```

---

## 📖 Usage Guide

### Basic Usage

```bash
# Run complete pipeline
python fcdt_master_script.py --mimic_path /data/mimic-iv-2.2/
```

### Advanced Options

#### Resume from Checkpoint

```bash
# Resume from feature engineering (if data extraction complete)
python fcdt_master_script.py \
    --mimic_path /data/mimic-iv-2.2/ \
    --resume features

# Resume from training (if features ready)
python fcdt_master_script.py \
    --mimic_path /data/mimic-iv-2.2/ \
    --resume training

# Generate figures only
python fcdt_master_script.py --figures_only
```

#### Memory Optimization

```bash
# Reduce batch size (for systems with <32GB RAM)
python fcdt_master_script.py \
    --mimic_path /data/mimic-iv-2.2/ \
    --batch_size 16

# Reduce epochs (for faster testing)
python fcdt_master_script.py \
    --mimic_path /data/mimic-iv-2.2/ \
    --num_epochs 25
```

#### Force CPU Usage

```bash
# Disable GPU
export CUDA_VISIBLE_DEVICES=""
python fcdt_master_script.py --mimic_path /data/mimic-iv-2.2/
```

#### Skip System Checks

```bash
# Skip RAM/disk space verification
python fcdt_master_script.py \
    --mimic_path /data/mimic-iv-2.2/ \
    --skip_check
```

---

## 🔬 Reproducing Paper Results

### Pipeline Stages

#### Stage 1: Data Extraction (~30-45 minutes)

Extracts T2DM cohort from MIMIC-IV:
- Identifies 8,127 T2DM patients using ICD codes
- Extracts labs, vitals, medications, outcomes
- Applies inclusion criteria (≥3 glucose measurements, ≥6 months follow-up)

```bash
python fcdt_tpff_data.py --mimic_path /data/mimic-iv-2.2/
```

#### Stage 2: Feature Engineering (~15-20 minutes)

Multi-scale temporal decomposition:
- Decomposes trajectories into micro/meso/macro scales
- Calculates TIR, eGFR, HOMA-IR, BMI
- Handles irregular sampling with time-gap encoding

```bash
python fcdt_tpff_features.py
```

#### Stage 3: Model Training (~45-60 minutes with GPU)

Trains complete FCDT-TPFF framework:
- Multi-scale LSTM encoding
- DTW-attention alignment  
- Graph attention networks
- Progression-aware clustering

```bash
python fcdt_tpff_training.py
```

#### Stage 4: Evaluation (~10-15 minutes)

Comprehensive evaluation:
- Compares against 6 baseline methods
- Computes clustering metrics (silhouette, Davies-Bouldin, C-H)
- Performs ablation study
- Analyzes treatment response associations

#### Stage 5: Figure Generation (~2-3 minutes)

Generates all paper figures ready for LaTeX insertion.

```bash
python fcdt_tpff_figures.py
```

### Generated Figures

All figures are publication-ready and saved in `./figures/`:

| Figure | Description | Paper Section |
|--------|-------------|---------------|
| `figure1_pipeline.png` | Framework architecture | Methods |
| `figure2_silhouette_comparison.png` | Clustering quality comparison | Results |
| `figure3_umap_clusters.png` | 2D UMAP visualization | Results |
| `figure4_kaplan_meier.png` | Survival curves by cluster | Results |
| `figure5_treatment_response.png` | Treatment response heatmap | Results |
| `figure6_ablation.png` | Ablation study results | Results |
| `figure7_missing_data.png` | Robustness to missingness | Results |

### LaTeX Integration

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.9\columnwidth]{figures/figure1_pipeline.png}
\caption{Framework pipeline showing four sequential stages...}
\label{fig:pipeline}
\end{figure}
```

---

## ⚠️ Troubleshooting

### Out of Memory Errors

**Problem:** `RuntimeError: CUDA out of memory` or system freezes

**Solutions:**

```bash
# Option 1: Reduce batch size
python fcdt_master_script.py \
    --mimic_path /data/mimic-iv-2.2/ \
    --batch_size 16

# Option 2: Force CPU usage
export CUDA_VISIBLE_DEVICES=""
python fcdt_master_script.py --mimic_path /data/mimic-iv-2.2/

# Option 3: Edit config directly
# In fcdt_master_script.py, change:
# config.BATCH_SIZE = 16
# config.MAX_PATIENTS_IN_MEMORY = 500
```

### Missing MIMIC-IV Files

**Problem:** `FileNotFoundError: [file] not found`

**Solution:** Ensure you have downloaded ALL required MIMIC-IV files:

```
Required files:
├── hosp/diagnoses_icd.csv.gz
├── hosp/patients.csv.gz
├── hosp/admissions.csv.gz
├── hosp/labevents.csv.gz
├── hosp/prescriptions.csv.gz
└── icu/chartevents.csv.gz
```

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'torch'`

**Solution:**

```bash
# Reinstall all dependencies
pip install -r requirements.txt --upgrade

# If GPU issues persist, install CPU version
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Slow Performance

**Problem:** Training takes >4 hours

**Solutions:**

1. **Enable GPU acceleration** (if available)
2. **Reduce epochs** for testing: `--num_epochs 25`
3. **Use smaller subset** (add `--debug_mode` flag if implemented)

### Figure Generation Fails

**Problem:** Figures are blank or corrupted

**Solution:**

```bash
# Ensure matplotlib backend is set correctly
export MPLBACKEND=Agg
python fcdt_tpff_figures.py
```

---

## 📊 Expected Performance

### Computational Requirements

| Stage | Time (GPU) | Time (CPU) | Peak RAM |
|-------|-----------|-----------|----------|
| Data Extraction | 30-45 min | 30-45 min | 8-12 GB |
| Feature Engineering | 15-20 min | 20-30 min | 12-16 GB |
| Model Training | 45-60 min | 3-4 hours | 16-24 GB |
| Evaluation | 10-15 min | 15-20 min | 8-12 GB |
| Figure Generation | 2-3 min | 2-3 min | 4-6 GB |
| **Total** | **~2-3 hours** | **~4-6 hours** | **32 GB** |

### Clustering Metrics

Expected results should match paper (within random seed variance):

- **Silhouette Score:** 0.41 ± 0.03
- **Davies-Bouldin Index:** 0.87 ± 0.08
- **C-statistic:** 0.78 (95% CI: 0.75-0.81)

If your results deviate significantly, check:
1. ✅ Correct MIMIC-IV version (v2.2)
2. ✅ Correct cohort size (N=8,127)
3. ✅ Random seed is fixed (seed=42)

---

## 📚 Citation

If you use this code or methodology in your research, please cite:

```bibtex
@inproceedings{maji2025trajectory,
  title={Graph-Based Trajectory Modeling for Progression-Aware Type 2 Diabetes Subtyping},
  author={Maji, Ankita and Vijayvergiya, Shashwat and Yadav, Vikas Kumar},
  booktitle={Proceedings of [IEEE Conference Name]},
  year={2025},
  organization={IEEE}
}
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**Important:** MIMIC-IV data usage is subject to:
- PhysioNet credentialing requirements
- CITI training completion
- Data use agreement compliance

---

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

##  Contact

**Authors:**
- **Ankita Maji** - [ankitamaji7033@gmail.com](mailto:ankitamaji7033@gmail.com)


**Institution:** Lovely Professional University, Phagwara, India

For issues and questions:
- 🐛 **Bug reports:** [Open an issue](https://github.com/Ankita7033/diabetes-subtyping/issues)
- 💬 **Discussions:** [GitHub Discussions](https://github.com/Ankita7033/diabetes-subtyping/discussions)

---

## 🙏 Acknowledgments

- **MIMIC-IV Team** at MIT Laboratory for Computational Physiology
- **PhysioNet** for data hosting and access infrastructure
- **Lovely Professional University** for computational resources
- **IEEE** for publication venue

---

## 🔄 Version History

### Version 1.0.0 (2025-01-XX)
- ✨ Initial public release
- ✅ Complete reproduction pipeline
- ✅ All paper figures and tables
- ✅ Optimized for 32GB RAM
- ✅ GPU and CPU support

---

## 📝 Notes

### Data Availability
- MIMIC-IV data: Available at [PhysioNet](https://physionet.org/content/mimiciv/2.2/)
- Processed features: Generated by pipeline (not shared due to privacy)
- Model weights: Included in repository after training

### Reproducibility
- Random seed: Fixed at 42
- PyTorch version: 2.0+
- Results may vary slightly (~1-2%) across different hardware/OS

### Known Limitations
- ICU population may not generalize to outpatient settings
- Requires substantial computational resources (32GB RAM)
- MIMIC-IV access requires credentialing (1-2 week process)

---

**⭐ If you find this work useful, please consider starring the repository!**

---