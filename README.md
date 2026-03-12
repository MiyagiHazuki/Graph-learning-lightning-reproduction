# ProGNN Reproduction

ProGNN aims to enhance the robustness of Graph Neural Networks against adversarial structural attacks by exploring the intrinsic properties of real-world graphs.

## Key Features

This project refactors the code structure by decoupling the graph structure learning module **`ProGNNLearner`** from complex training logic, making it a **pluggable independent component**. This allows developers to easily integrate it into existing GNN models or replace it with other graph learning strategies.

- **Independent Component**: The `ProGNNLearner` class in `src/models.py` encapsulates all core logic for graph structure learning, including low-rank, sparsity, feature smoothness constraints, and PGD optimization.
- **Flexible Configuration**: All hyperparameters are managed through YAML configuration files, making experimental settings easy to adjust.
- **Easy Integration**: Compatible with PyTorch Lightning's `GCNWrapper`, enabling ProGNN training mode with a single `graph_learner` parameter.

## Data Organization

Data is organized in three core directories:

```text
.
├── meta/                   # Global attack (Metattack) perturbed data
│   ├── {dataset}_meta_adj_{rate}.npz
│   └── ...
├── nettack/                # Targeted attack (Nettack) perturbed data
│   ├── {dataset}_nettack_adj_{perturbations}.npz
│   ├── {dataset}_nettacked_nodes.json
│   └── ...
└── splits/                 # Dataset splits (Train/Val/Test)
    ├── {dataset}_prognn_splits.json
    └── ...
```

- **meta**: Contains perturbed graph structures under different global perturbation rates from Metattack attacks. Files follow the format `{dataset}_meta_adj_{rate}.npz`, where `{rate}` represents the global perturbation ratio (proportion of modified edges to total edges). Used for testing robustness under global structural damage.
- **nettack**: Contains targeted attack data, including `{dataset}_nettack_adj_{perturbations}.npz` (damaged adjacency matrices, where `{perturbations}` is the number of edge modifications per target node) and `{dataset}_nettacked_nodes.json` (indices of attacked nodes). Used for evaluating classification stability under local node attacks.
- **splits**: Contains `{dataset}_prognn_splits.json` (fixed train/validation/test set split indices) to avoid performance evaluation bias from random splits and ensure fair model comparison.

## Quick Start

Training can be run directly through `main.py` with specified configuration files:

```bash
# Run Citeseer dataset experiment
python main.py --config config/citeseer.yaml

# Run Cora dataset experiment
python main.py --config config/cora.yaml
```

## Graph Learner Configuration Guide

The core improvement of this reproduction lies in the `ProGNNLearner` class in `src/models.py`. It serves as an independent module responsible for dynamically optimizing the adjacency matrix during training.

### 1. Instantiating Graph Learner

Initializing `ProGNNLearner` is straightforward, requiring only the initial adjacency matrix, configuration parameters, and computing device:

```python
from src.models import ProGNNLearner

# 1. Prepare initial adjacency matrix (must be Dense format [N, N])
# Typically use torch_geometric.utils.to_dense_adj for conversion
adj_dense = ... 

# 2. Initialize Learner
# args object needs to include symmetric, lr_adj, alpha, beta and other hyperparameters
graph_learner = ProGNNLearner(adj_dense, args, device)
```

### 2. Integration into GCN Model

Pass the instantiated `graph_learner` to `GCNWrapper`, and the model automatically switches to ProGNN's alternating optimization mode:

```python
from src.model_wrapper import GCNWrapper

model = GCNWrapper(
    nfeat=...,
    nhid=...,
    nclass=...,
    graph_learner=graph_learner,  # <--- Inject graph learning module
    args=args
)
```

If `graph_learner` is `None`, the model falls back to standard GCN training.

### 3. Key Configuration Parameters

The behavior of `ProGNNLearner` can be finely controlled through `config/*.yaml` files. Here are the key parameters affecting graph structure learning:

| Parameter | Description | Typical Value |
| :--- | :--- | :--- |
| `symmetric` | Whether to enforce symmetry in learned adjacency matrix | `false` |
| `lr_adj` | Learning rate for adjacency matrix optimization | `0.01` |
| `alpha` | **L1 norm coefficient**: Controls graph sparsity | `5e-4` |
| `beta` | **Nuclear norm coefficient**: Controls graph low-rank property | `1.5` |
| `gamma` | Weight for GCN classification loss | `1` |
| `lambda_` | Feature smoothness constraint weight | `0` (adjustable) |
| `phi` | Symmetry constraint weight | `0` (adjustable) |
| `outer_steps` | **Structure update steps**: Number of graph structure optimizations per epoch | `1` |
| `inner_steps` | **Parameter update steps**: Number of GCN parameter optimizations per epoch | `2` |

By adjusting these parameters, you can balance graph sparsity, low-rank properties, and feature smoothness to achieve optimal robustness under different levels of attacks.