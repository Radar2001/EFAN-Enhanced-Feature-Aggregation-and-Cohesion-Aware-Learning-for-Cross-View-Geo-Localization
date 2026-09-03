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

## 📦 Environment Setup

We recommend using a virtual environment with Python 3.8+ and PyTorch 1.9+.

```bash
# Clone the repository
git clone https://github.com/Radar2001/EFAN-Enhanced-Feature-Aggregation-and-Cohesion-Aware-Learning-for-Cross-View-Geo-Localization.git
cd EFAN

# Install dependencies
pip install -r requirements.txt


---
## 📈 Results

### University‑1652

| Method | R@1 (Drone→Sat) | AP (Drone→Sat) | R@1 (Sat→Drone) | AP (Sat→Drone) |
|--------|----------------|----------------|----------------|----------------|
| Ours (EFAN) | **96.03** | **96.72** | **97.15** | **95.79** |

### SUES‑200 (Drone→Satellite, R@1)

| Altitude | 150m | 200m | 250m | 300m |
|----------|------|------|------|------|
| Ours (EFAN) | **98.40** | **99.58** | **99.63** | **99.80** |

### DenseUAV (Drone→Satellite)

| Method | R@1 | R@5 | SDM@1 |
|--------|-----|-----|-------|
| Ours (EFAN) | **99.01** | **99.74** | **99.20** |
