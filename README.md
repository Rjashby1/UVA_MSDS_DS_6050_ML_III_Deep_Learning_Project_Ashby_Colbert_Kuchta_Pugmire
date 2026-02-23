# DS 6050 ML III Deep Learning Project — Skin Lesion Classification (ISIC 2019 + MILK10k)

This repository is for our DS 6050 (ML III: Deep Learning) project with Prof. Heman Shakeri, PhD. (UVA).  
At this stage, it reflects our **proposal + planned experimental pipeline** (results will be added as experiments complete).

**Team:** Robert Ashby, Xavier Colbert, Jacob Kuchta, Alysa Pugmire  
**Term:** Spring 2026

---

## Project Summary (Proposal)

Deep learning models for dermatology can look great on curated datasets but fail under real-world shifts (lighting, skin tone, artifacts, class imbalance).  
Our project is a **data-centric ablation study**: we hold architectures constant and systematically evaluate how different **augmentation / preprocessing strategies** affect generalization across datasets.

We benchmark three CNN families:
- ResNet
- EfficientNet
- DenseNet

---

## Datasets (Public, not stored in this repo)

We use two International Skin Imaging Collaboration (ISIC) - hosted public datasets:

- **ISIC 2019**: primary training + internal validation dataset  
- **MILK10k (ISIC 2025)**: external validation dataset (out-of-distribution generalization)

⚠️ **The raw images are NOT committed to GitHub** (too large).  
All links + citation notes are in: `DATA/DATA_Links.txt`

---

## One-time Data Setup

After cloning the repo, run:

```bash
python scripts/download_data.py
