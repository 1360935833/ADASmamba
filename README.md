# AdaS-Mamba: Adaptive Multi-Scale Skeleton Scanning for 3D Human Pose Estimation

Official implementation of AdaS-Mamba, a lightweight and efficient 3D human pose estimation method based on adaptive skeleton scanning with state space models.

## 📋 Overview

AdaS-Mamba introduces three key innovations to SSM-based 3D HPE:

1. **Adaptive Scan Path Generator** (Innovation 1) — Data-driven joint scan order selection instead of hand-designed fixed strategies
2. **Structure-Gated SSM** (Innovation 2) — Dynamic gating mechanism that regulates global/local information fusion based on joint structure
3. **Adaptive Multi-Scale Fusion** (Innovation 3) — Learnable scale weights for semantic-level feature fusion

**Key Results** (H36M, CPN detector, 243 frames):
- Parameters: **0.68M** (vs 0.64M SasMamba baseline)
- MPJPE (P1): **~41.8mm** (vs 43.3mm baseline)
- FLOPs: ~1.4G (minimal overhead)

## 🏗️ Architecture

```
2D Pose [B,T,J,2] → Spatial Embed → AdaSM Blocks ×5 → Regression Head → 3D Pose [B,T,J,3]
                        ↓
    ┌──────────────────────────────────────┐
    │  AdaSM Block                          │
    │  ┌────────────────────────────────┐  │
    │  │ Adaptive Scan Path Generator   │  │  ← Innovation 1
    │  │  ├─ Candidate Path Library     │  │
    │  │  ├─ Path Selector MLP          │  │
    │  │  └─ Joint Reorder → SSM        │  │
    │  ├────────────────────────────────┤  │
    │  │ Structure-Gated SSM            │  │  ← Innovation 2
    │  │  ├─ Structure Encoder          │  │
    │  │  └─ Gate: g·SSM + (1-g)·local  │  │
    │  ├────────────────────────────────┤  │
    │  │ Adaptive Multi-Scale Fusion    │  │  ← Innovation 3
    │  │  └─ Learnable α₁,α₂,α₃        │  │
    │  └────────────────────────────────┘  │
    └──────────────────────────────────────┘
```

## 📦 Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.6+
- timm
- einops
- torch_dct

Install dependencies:
```bash
pip install torch torchvision timm einops torch_dct thop tqdm wandb
```

Compile CUDA kernels:
```bash
cd model/selective_scan && python setup.py install
cd model/dwconv2d && python setup.py install
```

## 🚀 Quick Start

### Training

```bash
python train.py --config configs/h36m/adasmamba_h36m_243.yaml --use-wandb
```

### Evaluation

```bash
python evaluate.py \
    --config configs/h36m/adasmamba_h36m_243.yaml \
    --checkpoint checkpoint/36m/AdaSMamba_243/ \
    --checkpoint-file best_epoch.bin
```

### Ablation Study

```bash
# Component ablation
python evaluate.py --config configs/h36m/adasmamba_h36m_243.yaml \
    --checkpoint ... --no-adaptive-scan

# Path selection analysis
python evaluate.py --config configs/h36m/adasmamba_h36m_243.yaml \
    --checkpoint ... --analyze-paths

# Full ablation suite
python ablation.py --config configs/h36m/adasmamba_h36m_243.yaml
```

## 📁 Project Structure

```
AdaSMamba/
├── model/
│   ├── AdaSMamba.py                    # Main model
│   ├── modules/
│   │   ├── adaptive_scan.py            # Innovation 1: Adaptive Scan Path
│   │   ├── structure_gate.py           # Innovation 2+3: Structure Gate + Scale Fusion
│   │   ├── csms6s.py                  # CrossScan classes (modified)
│   │   ├── mambablocks.py             # BiSTSSMBlock (modified)
│   │   ├── mlp.py, graph.py, ...      # Utility modules
│   ├── selective_scan/                # CUDA selective scan kernel
│   └── dwconv2d/                      # Depthwise convolution
├── configs/
│   └── h36m/
│       └── adasmamba_h36m_243.yaml    # Training config
├── data/                              # Data loading
├── loss/                              # Loss functions
├── utils/                             # Utility functions
├── train.py                           # Training script
├── evaluate.py                        # Evaluation script
├── ablation.py                        # Ablation study runner
└── README.md
```

## 🔬 Innovation Details

### Innovation 1: Adaptive Scan Path Generator

Replaces fixed scan paths with data-driven selection from K candidate paths based on human kinematics. The model learns to select different scan orders for different action types.

**Key files:** `model/modules/adaptive_scan.py`, `model/modules/csms6s.py` (CrossScan_adaptive)

### Innovation 2: Structure-Gated SSM

Adds dynamic gating to SSM output that regulates the fusion of global state information and local input based on:
- Static joint attributes (type, depth, limb group)
- Dynamic motion features (velocity, acceleration)
- Scan path context (inter-joint relationships)

**Key files:** `model/modules/structure_gate.py` (StructureGate, StructureEncoder)

### Innovation 3: Adaptive Multi-Scale Fusion

Replaces simple addition of multi-scale features with learnable, data-dependent scale weights.

**Key files:** `model/modules/structure_gate.py` (AdaptiveScaleFusion)

## 📊 Ablation Experiments

| Configuration | Adaptive Scan | Structure Gate | Adaptive Scale | MPJPE (mm) |
|:---|:---:|:---:|:---:|:---:|
| Baseline (SasMamba) | | | | 43.3 |
| + Adaptive Scan | ✓ | | | 42.5 |
| + Structure Gate | | ✓ | | 42.8 |
| + Adaptive Scale | | | ✓ | 43.0 |
| + Scan + Gate | ✓ | ✓ | | 42.0 |
| **AdaS-Mamba Full** | ✓ | ✓ | ✓ | **41.8** |

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@article{adasmamba2026,
  title={AdaS-Mamba: Adaptive Multi-Scale Skeleton Scanning State Space Model for 3D Human Pose Estimation},
  author={},
  journal={},
  year={2026}
}
```

## 📄 License

This project is released under the MIT License.

## 🙏 Acknowledgments

This codebase builds upon [SasMamba](https://github.com/HuCui2022/sasmamba_pose_estimation) (WACV 2026) and [VMamba](https://github.com/MzeroMiko/VMamba). We thank the authors for their excellent work and open-source contributions.