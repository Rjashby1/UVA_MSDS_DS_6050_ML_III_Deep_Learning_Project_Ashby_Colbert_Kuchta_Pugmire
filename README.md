# Skin Lesion Classification  
## Data-centric augmentation ablation on CNNs (ISIC 2019 → MILK10k OOD)

This repo contains the code and documentation for our DS 6050 (ML III: Deep Learning) course project at the University of Virginia, supervised by Prof. Heman Shakeri, PhD.

**Project stage:** Proposal + planned experimental pipeline (results and training code will be added iteratively)  
**Team:** Robert Ashby, Xavier Colbert, Jacob Kuchta, Alysa Pugmire  
**Term:** Spring 2026

---

## Motivation and Problem Statement

Deep learning models for dermatological image classification can perform strongly on curated benchmarks yet degrade under real-world distribution shifts:

- illumination / color differences  
- varied skin tones  
- artifacts (hair, ink, markers)  
- severe class imbalance (rare malignant classes)

Rather than proposing a novel architecture, this project is **data-centric**:

> We hold architectures constant and systematically measure how augmentation / preprocessing choices affect robustness and generalization.

---

## Research Questions and Hypotheses

**RQ1 — Augmentation value**  
Which augmentation families provide the largest marginal gains over a strict baseline?

**RQ2 — Generalization under shift**  
Do synthetic augmentations improve out-of-distribution performance, or mainly inflate in-distribution validation?

**RQ3 — Metadata contribution**  
Does adding patient metadata (age/sex/anatomical site) via late fusion meaningfully improve performance?

**High-level hypothesis**  
Geometric and controlled color augmentations improve robustness to nuisance variation, and patch/mixing augmentations reduce reliance on spurious local cues—especially improving OOD performance.

---

## Datasets (public; not stored in this repo)

We use public datasets hosted by the International Skin Imaging Collaboration (ISIC):

- **ISIC 2019** — primary training + internal validation  
- **MILK10k (ISIC 2025)** — external validation / out-of-distribution (OOD) generalization test  

### Important
Raw images are **not committed** to GitHub (tens of GB).  
All official links and citation notes live in:

- `DATA/DATA_Links.txt`

---

## Quick start (one-time data setup)

### A) Clone the repo

    git clone <YOUR_REPO_URL>
    cd <YOUR_REPO_FOLDER>

### B) Create an environment + install dependencies

    python -m venv .venv

    # mac/linux
    source .venv/bin/activate

    # windows (PowerShell)
    # .venv\Scripts\Activate.ps1

    pip install -r requirements.txt

### C) Download and unzip datasets locally (ignored by git)

    python scripts/download_data.py

---

## Expected local data layout (ignored by git)

We standardize local paths so everyone’s environment matches the same structure:

    DATA/
    ├── raw/
    │   ├── isic2019/   (downloaded zips/csvs)
    │   └── milk10k/    (downloaded zips/csvs)
    ├── isic2019/
    │   ├── train/
    │   │   ├── images/
    │   │   ├── ISIC_2019_Training_Metadata.csv
    │   │   └── ISIC_2019_Training_GroundTruth.csv
    │   └── test/
    │       ├── images/
    │       ├── ISIC_2019_Test_Metadata.csv
    │       └── ISIC_2019_Test_GroundTruth.csv
    └── milk10k/
        ├── train/
        │   ├── images/
        │   ├── MILK10k_Training_Metadata.csv
        │   ├── MILK10k_Training_Supplement.csv
        │   └── MILK10k_Training_GroundTruth.csv
        └── test/
            ├── images/
            └── MILK10k_Test_Metadata.csv

---

## Proposed Methodology (ablation study)

### Backbones (held constant)

We benchmark three established CNN families:

- ResNet  
- EfficientNet  
- DenseNet  

### Late-fusion metadata branch (on/off condition)

In addition to image-only baselines, we evaluate an optional multi-modal variant using:

- age (normalized)  
- sex (one-hot)  
- anatomical site (one-hot)

Planned fusion:

1) CNN backbone → pooled feature vector  
2) metadata → small MLP → embedding  
3) concatenate (late fusion) → classifier head  

### Four-phase augmentation ladder (main ablation axis)

**Phase 1 — Strict baseline (no augmentation)**  
- center crop / resize / normalization only

**Phase 2 — Geometric / affine transforms**  
- flips, constrained rotations, mild shear/affine

**Phase 3 — Color / illumination augmentation**  
- brightness/contrast/hue/saturation perturbations (controlled)

**Phase 4 — Scaling + patch/mixing augmentation**  
- random zoom/crops + patchwise masking/mixing (e.g., CutMix-style)

### Class imbalance handling

ISIC 2019 is imbalanced. We plan to use **weighted cross-entropy** with class weights inversely proportional to class frequency.

### Cross-dataset generalization protocol

- Train + internal validation: **ISIC 2019**  
- Final OOD evaluation: **MILK10k**

This isolates whether augmentations improve real robustness vs. only internal validation.

---

## Evaluation Metrics

Because accuracy can be misleading under imbalance, we emphasize:

- ROC-AUC  
- Balanced Accuracy (BACC)  
- Sensitivity / Recall (especially for malignant classes)

---

## Experiment Plan (Ablation Matrix)

We will run a controlled ablation across:

- **Backbone:** ResNet / EfficientNet / DenseNet  
- **Augmentation phase:** 1 → 4  
- **Metadata fusion:** off vs on  

A results table will be populated as experiments complete:

| Backbone | Aug Phase | Metadata | ISIC 2019 (AUC/BACC) | MILK10k OOD (AUC/BACC) |
|---|---:|---:|---|---|
| ResNet | 1 | off | TBD | TBD |
| ResNet | 2 | off | TBD | TBD |
| ... | ... | ... | ... | ... |

---

## Repo Hygiene (please don’t commit big files)

This repo is for code, configs, and lightweight outputs.

Do **not** commit:
- `DATA/` (raw or extracted images)
- large checkpoints (`*.pt`, `*.pth`, `*.ckpt`)
- large logs/artifacts

---

## Citations / Data Attribution

Dataset links + citation notes:
- `DATA/DATA_Links.txt`

MILK10k citation:

MILK study team. *MILK10k*. ISIC Archive, 2025. doi:10.34970/648456.

ISIC 2019 citation:

[1] Tschandl P., Rosendahl C. & Kittler H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 doi.10.1038/sdata.2018.161 (2018)

[2] Noel C. F. Codella, David Gutman, M. Emre Celebi, Brian Helba, Michael A. Marchetti, Stephen W. Dusza, Aadi Kalloo, Konstantinos Liopyris, Nabin Mishra, Harald Kittler, Allan Halpern: "Skin Lesion Analysis Toward Melanoma Detection: A Challenge at the 2017 International Symposium on Biomedical Imaging (ISBI), Hosted by the International Skin Imaging Collaboration (ISIC)", 2017; arXiv:1710.05006.

[3] Hernández-Pérez C, Combalia M, Podlipnik S, Codella NC, Rotemberg V, Halpern AC, Reiter O, Carrera C, Barreiro A, Helba B, Puig S, Vilaplana V, Malvehy J. BCN20000: Dermoscopic lesions in the wild. Scientific Data. 2024 Jun 17;11(1):641.

---

## Disclaimer

This project is for academic research in DS 6050 and is not a clinical device.  
No medical advice is provided or implied.
