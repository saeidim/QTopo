# QTopo-Meta

**Quantum-Inspired Topological Meta-Learning for Cross-Domain Few-Shot Pattern Recognition**

Saeid Iranmanesh · Raad Raad  
University of Wollongong

---

## Overview

QTopo-Meta is a cross-modal few-shot learning framework that combines:

- **Stage 1** — Modality-to-point-cloud conversion (Takens delay embedding for signals, Gaussian expansion for images/tabular)
- **Stage 2** — Quantum-inspired Random Fourier Feature encoding (shift-invariant kernel approximation, Lipschitz-bounded, √(2/D) normalization)
- **Stage 3** — Persistent homology + persistence-image vectorization (Vietoris–Rips filtration, sliced-Wasserstein distance)
- **Stage 4** — Topology-aware prototypical meta-learner (FOMAML, learnable Euclidean–Wasserstein blend)
- **Stage 5** — Few-shot inference with test-time adaptation

The central finding is that **persistent homology drives cross-modal transfer**: lifting heterogeneous modalities to a shared topological representation reduces domain divergence by ~33% and enables cross-modal few-shot classification well above chance (37.5% vs. 20% on the 5-way 5-shot CD-3 benchmark).

---

## Installation

```bash
git clone https://github.com/saeidim/QTopo.git
cd QTopo
pip install -r requirements.txt
```

GPU training requires a CUDA-enabled PyTorch build; see
https://pytorch.org/get-started/locally/ for the right install command.

---

## Quick Start

```bash
# Smoke-test (5 epochs, fast)
python main.py --quick

# Full run: UCI-HAR raw + Indian Pines, 1-shot and 5-shot
python main.py --datasets uci_har_raw indian_pines --shots 1 5

# Include cross-domain evaluation
python main.py --datasets uci_har_raw indian_pines --shots 1 5 --cross_domain

# Heterogeneous episodic training (QTopo-Meta variant)
python main.py --datasets uci_har_raw indian_pines --shots 5 \
               --cross_domain --hetero_train

# Compare a subset of methods
python main.py --methods ProtoNet MAML QTopo_Meta --cross_domain
```

Results are written to `results/` as CSV files. Plots go to `plots/`.

---

## Repository Layout

```
QTopo/
├── config.py                  # All hyperparameters (edit here)
├── main.py                    # Experiment runner
├── requirements.txt
├── datasets/
│   ├── loaders.py             # Dataset download + class splitting
│   └── tda_pipeline.py        # Stages 1–3: point cloud → TDA features
├── models/
│   └── all_models.py          # All models + EpisodeSampler
├── experiments/
│   └── trainer.py             # MetaTrainer + final_test
└── utils/
    └── plotting.py            # Training curves + result tables
```

---

## Key Hyperparameters

Defined in `config.py`. The most impactful:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `n_qubits` | 6 | PCA target dim (Stage 2). Values >6 inflate Lipschitz constant with no benefit |
| `r` | 40 | Landmark count for maxmin subsampling (Stage 3). Values >50 produce degenerate complexes on compressed clouds |
| `n_fit_cap` | 1000 | Max samples for PCA fit. Controls output dim consistency across domains |
| `D` | 512 | Random Fourier feature dimension |
| `max_dim` | 1 | Max homology dimension (H₀+H₁; H₂ is ~100× slower) |

---

## Reproducing the Paper's Case Study (CD-3 Benchmark)

The CD-3 benchmark uses two modalities: UCI-HAR raw (inertial signal) and Indian Pines (hyperspectral signal). Both datasets are downloaded automatically on first run.

```bash
python main.py \
  --datasets uci_har_raw indian_pines \
  --shots 1 5 \
  --methods ProtoNet MAML RelationNet FWT TIN_FSL QTopo_Meta \
  --cross_domain \
  --seed 42
```

Expected results (5-way, 5-shot, cross-modal average):

| Method | Seen-within | Unseen-within | Cross-modal |
|--------|-------------|---------------|-------------|
| ProtoNet | 72.1% | 51.2% | 28.4% |
| TDA ProtoNet | — | — | **37.5%** |
| QTopo-Meta | 73.6% | 62.8% | 33.1% |

> **Note**: TDA ProtoNet (QTopo-Meta without Stage 2 quantum encoder) achieves the best cross-modal accuracy, showing that persistent homology is the primary driver of cross-domain transfer. The quantum encoder adds noise robustness (see ablation E1 in the paper).

---

## Citation

If you use this code or the CD-3 benchmark, please cite:

```bibtex
@article{iranmanesh2026qtopometa,
  title   = {{QTopo-Meta}: Quantum-Inspired Topological Meta-Learning
             for Cross-Domain Few-Shot Pattern Recognition},
  author  = {Iranmanesh, Saeid and Raad, Raad},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

MIT License — see `LICENSE` for details.

The code and data that reproduce the case study are available at  
https://github.com/saeidim/QTopo
