"""
config.py — Central configuration for QTopo-Meta experiments.
Edit this file to change hyperparameters, dataset paths, and experiment settings.

Repository: https://github.com/saeidim/QTopo
Paper: QTopo-Meta: Quantum-Inspired Topological Meta-Learning for
       Cross-Domain Few-Shot Pattern Recognition
Authors: Saeid Iranmanesh, Raad Raad — University of Wollongong
"""

import os
import torch

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════

ROOT_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(ROOT_DIR, "data")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
PLOTS_DIR   = os.path.join(ROOT_DIR, "plots")
CKPT_DIR    = os.path.join(ROOT_DIR, "checkpoints")

for d in [DATA_DIR, RESULTS_DIR, PLOTS_DIR, CKPT_DIR]:
    os.makedirs(d, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# DEVICE
# ══════════════════════════════════════════════════════════════════════════════

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[config] Using device: {DEVICE}")

# ══════════════════════════════════════════════════════════════════════════════
# DATASET SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

DATASET_CFG = {
    "uci_har": {
        "modality"    : "signal",
        "n_classes"   : 6,
        "n_train_cls" : 4,
        "n_val_cls"   : 1,
        "n_test_cls"  : 1,
        "download_url": "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
        # If download fails, sklearn fallback used (OpenML id=1478)
    },
    "uci_har_raw": {
        "modality"    : "signal",
        "n_classes"   : 6,
        "n_train_cls" : 4,
        "n_val_cls"   : 1,
        "n_test_cls"  : 1,
        # 128-timestep windows x 9 inertial channels — real signal topology
    },
    "covtype": {
        "modality"    : "tabular",
        "n_classes"   : 7,
        "n_train_cls" : 4,
        "n_val_cls"   : 1,
        "n_test_cls"  : 2,
    },
    "indian_pines": {
        # Modality is "signal" not "image": the 200-band spectral curve is
        # treated as a 1-D signal so Takens embedding extracts real topology.
        # image_to_cloud (Gaussian-ball expansion) is degenerate per pixel.
        "modality"    : "signal",
        "n_classes"   : 16,
        "n_train_cls" : 10,
        "n_val_cls"   : 3,
        "n_test_cls"  : 3,
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE HYPERPARAMETERS (Stages 1–3)
# ══════════════════════════════════════════════════════════════════════════════

PIPELINE_CFG = {
    # Stage 2: Quantum-inspired RFF encoding
    # n_qubits = 6: PCA target dimension.  Values > 6 inflate the Lipschitz
    # constant L_κ ∝ ‖A‖₂ with no topological benefit (eigenvectors beyond
    # the 6th carry noise for the HAR and Indian-Pines point clouds).
    "n_qubits"   : 6,

    "n_layers"   : 4,       # VQC depth (≤ 6 per PAC-Bayes bound)
    "D"          : 512,     # random Fourier feature dimension
    "sigma"      : 1.0,     # RFF kernel bandwidth

    # Stage 3: Persistent homology
    # r = 40: landmark count for maxmin subsampling.  r = 100 (old default)
    # exceeds the typical cloud size after PCA compression and causes ripser
    # to attempt Vietoris–Rips on a degenerate complex, producing spurious
    # H₁/H₂ classes.  r = 40 keeps the complex well-conditioned.
    "r"          : 40,

    "T"          : 20,      # persistence image grid resolution (T × T grid)
    "d"          : 256,     # output feature dimension (capped by n_fit – 1)
    "max_dim"    : 1,       # max homology dim (H₀ + H₁; H₂ is ~100× slower)
    "n_slices"   : 50,      # sliced-Wasserstein random projections

    # n_fit_cap = 1000: upper bound on samples used to fit the PCA projector.
    # The actual n_fit used is min(n_fit_cap, smallest_training_pool_size) so
    # the PCA output dimension d is identical across datasets — essential for
    # cross-domain evaluation without zero-padding or per-domain projectors.
    # Old value 200 capped d at 199, preventing the full 256-dim embedding.
    "n_fit_cap"  : 1000,
}

# ══════════════════════════════════════════════════════════════════════════════
# EPISODE SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

EPISODE_CFG = {
    "N"  : 5,    # ways  (N-way classification)
    "K"  : 5,    # shots (K support samples per class)
    "Q"  : 15,   # queries per class
}

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_CFG = {
    "n_epochs"           : 100,
    "episodes_per_epoch" : 100,
    "val_episodes"       : 200,
    "test_episodes"      : 1000,
    "patience"           : 20,    # early-stopping patience (epochs)
    "seed"               : 42,

    # Cosine annealing LR schedule: lr decays from lr_max to lr_min over
    # n_epochs × episodes_per_epoch steps, with T_max = n_epochs.
    "use_cosine_lr"      : True,
    "lr_min"             : 1e-5,  # cosine annealing floor
}

# ══════════════════════════════════════════════════════════════════════════════
# MODEL HYPERPARAMETERS (shared backbone)
# ══════════════════════════════════════════════════════════════════════════════

MODEL_CFG = {
    "d_prime"  : 128,    # embedding output dimension
    "hidden"   : 512,    # hidden layer width
    "dropout"  : 0.1,
    "lr"       : 1e-3,
    "wd"       : 1e-4,
    "eta"      : 0.01,   # MAML inner-loop learning rate
    "n_inner"  : 1,      # MAML inner adaptation steps

    # QTopo-Meta specific
    "alpha"    : 0.6,    # initial Euclidean–Wasserstein blend (learnable via sigmoid)
    "lambda1"  : 0.1,    # topological consistency regularizer weight
    "lambda2"  : 5e-4,   # L2 weight decay

    # RelationNet specific
    "relation_hidden": 256,

    # FWT specific
    "fwt_n_transforms": 8,
}

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

EXPERIMENT_CFG = {
    # Which datasets to run.  Cross-domain demo uses raw HAR (signal) +
    # Indian Pines (spectral).  CovType is excluded: per-row tabular data
    # has no class-discriminative topology and gives chance-level accuracy.
    "datasets"         : ["uci_har_raw", "indian_pines"],

    # Which methods to compare
    "methods"          : [
        "ProtoNet",
        "MAML",
        "RelationNet",
        "FWT",
        "TIN_FSL",
        "QTopo_Meta",
    ],

    # Evaluation settings
    "shots"            : [1, 5],   # 1-shot and 5-shot
    "cross_domain"     : True,     # run cross-domain evaluation
    "ablation"         : True,     # run ablation studies (E1–E8)
    "n_bootstrap"      : 600,      # bootstrap resamples for 95% CI
}
