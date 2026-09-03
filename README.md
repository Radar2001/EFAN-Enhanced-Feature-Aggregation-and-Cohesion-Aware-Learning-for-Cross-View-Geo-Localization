# EFAN: Enhanced Feature Aggregation and Cohesion-Aware Learning for Cross-View Geo-Localization

[![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg)](https://arxiv.org/abs/xxxx.xxxxx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.9+](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)

This repository is the official implementation of **EFAN**, a novel deep learning framework for **Cross-View Geo-Localization (CVGL)**. Our method achieves state-of-the-art performance on University-1652, SUES-200, and DenseUAV datasets.

> **Paper**: *Enhanced Feature Aggregation and Cohesion-Aware Learning for Cross-View Geo-Localization*  
> Huatao Yu, Shenao Du, Anxi Yu, Wenhao Tong, Zhen Dong

## ✨ Highlights

- **Scale-Unified Semantic-Spatial Enhancement Module (S³EM)** – synergistically suppresses semantic drift and enhances spatial context via parallel guided paths.
- **Complementary Feature Aggregation (CFA)** – explicitly fuses robust global semantics from the backbone with discriminative local details from S³EM.
- **Cohesion-Aware Adaptive Hybrid Loss (CAAHL)** – dynamically balances InfoNCE and Circle losses based on batch-wise feature cohesion, adapting optimization to the current feature space state.
- **State-of-the-art performance** – achieves **R@1 > 99%** on the challenging DenseUAV benchmark and outperforms all existing methods on University‑1652 and SUES‑200.

## 📈 Results

We compare EFAN with state‑of‑the‑art methods on University‑1652 and also evaluate its cross‑domain generalization by training on University‑1652 and testing directly on SUES‑200. **Bold** numbers indicate the best performance.

### University‑1652

| Method | R@1 (Drone→Sat) | AP (Drone→Sat) | R@1 (Sat→Drone) | AP (Sat→Drone) |
|--------|----------------|----------------|----------------|----------------|
| Instance Loss | 58.49 | 63.14 | 71.18 | 58.74 |
| LPN | 75.93 | 79.14 | 86.45 | 74.79 |
| RK‑Net | 77.60 | 80.55 | 86.59 | 75.96 |
| FSRA | 82.25 | 84.82 | 87.87 | 81.53 |
| TransFG | 84.01 | 86.31 | 90.16 | 84.61 |
| GeoFormer | 89.08 | 90.83 | 92.30 | 88.54 |
| MCCG | 89.64 | 91.32 | 94.30 | 89.39 |
| SDPL | 90.16 | 91.64 | 93.58 | 89.45 |
| Sample4Geo | 92.65 | 93.81 | 96.43 | 93.79 |
| SRLN | 92.70 | 93.77 | 95.14 | 91.97 |
| MFRGN | 94.33 | 95.24 | 96.15 | 93.94 |
| CAMP | 94.46 | 95.38 | 96.15 | 92.72 |
| DAC | 94.67 | 95.50 | 96.43 | 93.79 |
| ViT‑SegMatchNet | 92.60 | 93.80 | 95.59 | 92.30 |
| SURFNet | 94.57 | 95.49 | 95.72 | 93.20 |
| ECSNet | 94.80 | 95.68 | 96.57 | 92.89 |
| **Ours (EFAN)** | **96.03** | **96.72** | **97.15** | **95.79** |

### Cross‑Domain Generalization: University‑1652 → SUES‑200

Models are trained on University‑1652 and directly tested on SUES‑200 without fine‑tuning. Results are reported for both Drone‑to‑Satellite and Satellite‑to‑Drone tasks at different drone altitudes.

#### Drone → Satellite

| Method | 150m R@1 | 150m AP | 200m R@1 | 200m AP | 250m R@1 | 250m AP | 300m R@1 | 300m AP | Average R@1 |
|--------|----------|---------|----------|---------|----------|---------|----------|---------|-------------|
| Sample4Geo | 70.05 | 74.93 | 80.68 | 83.90 | 87.35 | 89.72 | 90.03 | 91.91 | 82.02 |
| CAMP | 78.90 | 82.38 | 86.83 | 89.28 | 91.95 | 93.63 | 95.68 | 96.65 | 88.34 |
| DAC | 76.65 | 80.56 | 86.45 | 89.00 | 92.95 | 94.18 | 94.53 | 95.45 | 87.65 |
| SURFNet | 86.10 | 88.22 | 91.88 | 93.20 | 95.07 | 95.83 | 96.00 | 96.55 | 92.26 |
| ECSNet | 82.85 | 88.02 | 89.21 | 91.00 | 92.89 | 94.20 | 94.66 | 95.80 | 89.90 |
| **Ours (EFAN)** | **88.38** | **90.31** | **94.30** | **95.23** | **96.45** | **97.09** | **97.35** | **97.79** | **94.12** |

#### Satellite → Drone

| Method | 150m R@1 | 150m AP | 200m R@1 | 200m AP | 250m R@1 | 250m AP | 300m R@1 | 300m AP | Average R@1 |
|--------|----------|---------|----------|---------|----------|---------|----------|---------|-------------|
| Sample4Geo | 83.75 | 73.83 | 91.25 | 83.42 | 93.75 | 89.07 | 93.75 | 90.66 | 90.63 |
| CAMP | 87.50 | 78.98 | 95.00 | 87.05 | 95.00 | 91.05 | 96.25 | 93.44 | 93.44 |
| DAC | 87.50 | 79.87 | 96.25 | 88.98 | 95.00 | 92.81 | 96.25 | 94.00 | 93.75 |
| SURFNet | 91.25 | 86.67 | 96.25 | 92.23 | 97.50 | 95.03 | 98.75 | 95.59 | 95.94 |
| ECSNet | 91.28 | 81.52 | 96.28 | 89.83 | 95.00 | 92.82 | 96.25 | 94.36 | 94.70 |
| **Ours (EFAN)** | **93.75** | **86.95** | **97.50** | **93.37** | **97.50** | **96.26** | **97.50** | **96.74** | **96.56** |

## 📦 Environment Setup

We recommend using a virtual environment with Python 3.8+ and PyTorch 1.9+.

```bash
# Clone the repository
git clone https://github.com/Radar2001/EFAN-Enhanced-Feature-Aggregation-and-Cohesion-Aware-Learning-for-Cross-View-Geo-Localization.git
cd EFAN

# Install dependencies
pip install -r requirements.txt




