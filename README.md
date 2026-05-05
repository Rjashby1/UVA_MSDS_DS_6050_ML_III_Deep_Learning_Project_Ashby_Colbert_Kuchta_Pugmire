# Skin Lesion Classification

## Data-centric augmentation ablation on CNNs, ISIC 2019 to MILK10k external evaluation

This repository contains the code, notebooks, helper scripts, and lightweight outputs for our DS 6050, ML III: Deep Learning course project at the University of Virginia, supervised by Prof. Heman Shakeri, PhD.

**Project stage:** Final project completed  
**Team:** Robert Ashby, Xavier Colbert, Jacob Kuchta, Alysa Pugmire  
**Term:** Spring 2026  

---

## Project Summary

This project evaluates whether data-centric augmentation strategies improve external robustness in skin lesion classification. Rather than proposing a novel neural architecture, we hold the CNN backbone family fixed and systematically vary the image augmentation policy.

The core experiment is a cumulative augmentation ladder evaluated across three ImageNet-pretrained CNN backbones:

- ResNet-50
- EfficientNet-B0
- DenseNet-121

The development dataset is ISIC 2019. MILK10k is reserved as an external-only evaluation cohort to test whether improvements persist under cross-dataset distribution shift.

The final image-only experiment includes 12 primary runs:

```text
3 CNN backbones x 4 cumulative augmentation stages = 12 runs
```

Metadata fusion was studied and showed minimal performance gains in this application.

---

## Key Finding

The strongest external result came from EfficientNet-B0 at Phase 1+2+3, which combines the strict baseline with geometric and color augmentation.

| Best model | Stage | ISIC BACC | MILK10k BACC | BACC Gap |
|---|---|---:|---:|---:|
| EfficientNet-B0 | Phase 1+2+3 | 0.759 | 0.571 | 0.187 |

The main result is that moderate augmentation improves external robustness, but the benefit is not monotonic. Phase 4, which added scale-cropping and CutMix, reduced external balanced accuracy across all three backbones.

---

## Research Questions

**RQ1, Augmentation value**  
Which augmentation families provide the largest marginal gains over a strict baseline?

**RQ2, Generalization under shift**  
Do augmentation gains persist under cross-dataset distribution shift from ISIC 2019 to MILK10k?

**RQ3, Metadata contribution**  
If compatible metadata fields are available across datasets, can late-fusion metadata provide additional predictive value beyond image-only learning?

In the final project, RQ1 and RQ2 are answered through the 12-run image-only augmentation study.  RQ3 is addressed through a metadata fusion experiment using EfficientNet-B0, which showed marginal gains only at an increased epoch budget and was excluded from the external evaluation.

---

## Datasets

This project uses public datasets hosted by the International Skin Imaging Collaboration, ISIC.

- **ISIC 2019**, development dataset for training and internal validation
- **MILK10k**, external-only evaluation cohort for out-of-distribution robustness testing

Raw images are not committed to GitHub because they are large. Dataset links and citation notes are stored in:

```text
DATA/DATA_Links.txt
```

---

## Final Augmentation Ladder

The augmentation study is cumulative rather than factorial. Each stage adds one transform family to the previous stage.

| Stage | Description |
|---|---|
| Phase 1 | Strict baseline, resize and normalization only |
| Phase 1+2 | Adds geometric transforms, including flips, constrained rotations, and mild affine variation |
| Phase 1+2+3 | Adds color-space perturbations, including brightness, contrast, saturation, and hue jitter |
| Phase 1+2+3+4 | Adds scale/crop and patch-based mixing, including RandomResizedCrop and CutMix |

This design allows each phase to be interpreted as an added augmentation family rather than an unrelated configuration.

---

## Final Results Summary

| Backbone | Stage | ISIC AUC | ISIC BACC | MILK AUC | MILK BACC | BACC Gap |
|---|---|---:|---:|---:|---:|---:|
| DenseNet-121 | P1 | 0.956 | 0.742 | 0.847 | 0.485 | 0.257 |
| DenseNet-121 | P1+2 | 0.955 | 0.737 | 0.866 | 0.542 | 0.194 |
| DenseNet-121 | P1+2+3 | 0.953 | 0.712 | 0.860 | 0.516 | 0.196 |
| DenseNet-121 | P1+2+3+4 | 0.940 | 0.697 | 0.846 | 0.441 | 0.256 |
| EfficientNet-B0 | P1 | 0.963 | 0.770 | 0.861 | 0.529 | 0.242 |
| EfficientNet-B0 | P1+2 | 0.963 | 0.774 | 0.866 | 0.533 | 0.242 |
| EfficientNet-B0 | P1+2+3 | 0.960 | 0.759 | 0.881 | 0.571 | 0.187 |
| EfficientNet-B0 | P1+2+3+4 | 0.943 | 0.727 | 0.861 | 0.516 | 0.211 |
| ResNet-50 | P1 | 0.956 | 0.702 | 0.819 | 0.437 | 0.265 |
| ResNet-50 | P1+2 | 0.959 | 0.748 | 0.835 | 0.481 | 0.267 |
| ResNet-50 | P1+2+3 | 0.955 | 0.732 | 0.857 | 0.519 | 0.214 |
| ResNet-50 | P1+2+3+4 | 0.947 | 0.711 | 0.842 | 0.454 | 0.257 |

**Interpretation:** Geometric and color augmentation improved external robustness most clearly. Phase 4 over-regularized the models and reduced MILK10k balanced accuracy for every backbone.

---

## Repository Structure

```text
.
├── Code/
│   ├── 00_project_setup.ipynb
│   ├── 01_data_audit_and_splits.ipynb
│   ├── 02_baseline_modeling.ipynb
│   ├── 03_augmentation_experiments.ipynb
│   ├── 04_ablation_study.ipynb
│   ├── 05_metadata_fusion_optional.ipynb
│   ├── 06_external_evaluation.ipynb
│   └── 07_results_synthesis.ipynb
├── DATA/
│   └── DATA_Links.txt
├── Data_Links/
├── outputs/
│   ├── configs/
│   ├── figures/
│   ├── metrics/
│   ├── preds/
│   ├── reports/
│   └── tables/
├── scripts/
├── environment.yml
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Notebook Workflow

The project is notebook-led. The main workflow is:

| Notebook | Purpose | Final status |
|---|---|---|
| `00_project_setup.ipynb` | Environment, paths, package checks, and project setup | Complete |
| `01_data_audit_and_splits.ipynb` | Data audit, split creation, class distributions, and metadata checks | Complete |
| `02_baseline_modeling.ipynb` | Phase 1 baseline training across three CNN backbones | Complete |
| `03_augmentation_experiments.ipynb` | Augmentation smoke tests and transform verification | Complete |
| `04_ablation_study.ipynb` | Full 12-run cumulative augmentation ablation on ISIC 2019 | Complete |
| `05_metadata_fusion_optional.ipynb` | Metadata Late Fusion exploration | Complete |
| `06_external_evaluation.ipynb` | MILK10k external evaluation of trained checkpoints | Complete |
| `07_results_synthesis.ipynb` | Final result aggregation, table generation, and report-ready figures | Complete |

Recommended review order:

```text
00 -> 01 -> 02 -> 03 -> 04 -> 06 -> 07
```

---

## Helper Scripts

The `scripts/` folder contains reusable model and augmentation code used by the notebooks. Key helper modules include:

```text
scripts/
├── resnet50_baseline.py
├── efficientnet_b0_baseline.py
├── densenet121_baseline.py
├── phase2_geometric.py
├── phase3_color.py
└── phase4_scale_crop.py
```

The model scripts define CNN builders and classifier heads. The augmentation scripts define the cumulative transform families used in the ablation ladder.

---

## Outputs

Lightweight outputs are stored under `outputs/`.

Important output locations:

```text
outputs/configs/     experiment configuration files
outputs/figures/     report and presentation figures
outputs/metrics/     CSV metric summaries and comparison tables
outputs/preds/       saved model predictions
outputs/reports/     classification reports and supporting summaries
outputs/tables/      report-ready tables
```

Large raw datasets and large model checkpoints are not intended to be committed.

---

## Evaluation Metrics

Because ISIC 2019 is highly imbalanced, plain accuracy is not emphasized. The primary metrics are:

- Macro-AUC
- Balanced Accuracy, BACC
- Per-class recall and confusion matrices
- BACC generalization gap

The generalization gap is defined as:

```text
BACC Gap = ISIC validation BACC - MILK10k external BACC
```

A smaller gap indicates better preservation of performance under external distribution shift.

---

## Environment Setup

### Option A, Conda or Mamba

```bash
mamba env create -f environment.yml
mamba activate ds6050-skin
```

If using Conda instead of Mamba:

```bash
conda env create -f environment.yml
conda activate ds6050-skin
```

### Option B, pip

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

---

## Data Setup

Raw data are not stored in GitHub. Download ISIC 2019 and MILK10k from ISIC using the links listed in:

```text
DATA/DATA_Links.txt
```

Expected local data layout:

```text
DATA/
├── raw/
│   ├── isic2019/
│   └── milk10k/
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
```

Processed split files and derived artifacts are generated by the notebooks and scripts.

---

## Reproducibility Notes

The final study uses a fixed training protocol across the 12 primary runs:

- Input size: 224 x 224
- ImageNet pretrained CNN backbones
- Optimizer: Adam
- Learning rate: 1e-4
- Weight decay: 1e-4
- Batch size: 32
- Epochs: 5
- Loss: weighted cross-entropy
- Model selection: best checkpoint by ISIC validation BACC
- External evaluation: MILK10k only after training and model selection

MILK10k was never used during training or checkpoint selection.

---

## Repo Hygiene

This repository is for code, configuration files, reports, and lightweight outputs.

Do not commit:

```text
DATA/
*.pt
*.pth
*.ckpt
large logs
large raw image folders
large intermediate artifacts
```

The `.gitignore` should keep raw data and large model artifacts out of version control.

---

## Final Report

The final report summarizes the methodology, results, external-evaluation findings, and future work.

Main report claim:

```text
Moderate augmentation, especially geometric and color augmentation, improved external robustness.
High-complexity Phase 4 augmentation degraded performance across all three CNN backbones.
EfficientNet-B0 at Phase 1+2+3 was the strongest external model.
```

---

## Citations and Data Attribution

Dataset links and citation notes are stored in:

```text
DATA/DATA_Links.txt
```

MILK10k citation:

```text
MILK study team. MILK10k. ISIC Archive, 2025. doi:10.34970/648456.
```

ISIC 2019 citation notes:

```text
Tschandl P., Rosendahl C. & Kittler H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161. doi:10.1038/sdata.2018.161. 2018.

Noel C. F. Codella, David Gutman, M. Emre Celebi, Brian Helba, Michael A. Marchetti, Stephen W. Dusza, Aadi Kalloo, Konstantinos Liopyris, Nabin Mishra, Harald Kittler, Allan Halpern. Skin Lesion Analysis Toward Melanoma Detection: A Challenge at the 2017 International Symposium on Biomedical Imaging, hosted by the International Skin Imaging Collaboration. arXiv:1710.05006. 2017.

Hernández-Pérez C., Combalia M., Podlipnik S., Codella N. C., Rotemberg V., Halpern A. C., Reiter O., Carrera C., Barreiro A., Helba B., Puig S., Vilaplana V., Malvehy J. BCN20000: Dermoscopic lesions in the wild. Scientific Data. 2024.
```

---

## Disclaimer

This project is for academic research in DS 6050 and is not a clinical device. No medical advice is provided or implied.