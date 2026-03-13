# ProGNN & ELR-GCN Reproduction

ProGNN and ELR-GCN aim to enhance the robustness of Graph Neural Networks against adversarial structural attacks by exploring the intrinsic properties of real-world graphs and using efficient low-rank estimations.

## Key Features

This project refactors the code structure by decoupling graph structure learning modules (**`ProGNNLearner`** and **`ELRGNNLearner`**) from complex training logic, making them **pluggable independent components**. This allows developers to easily integrate them into existing GNN models or replace them with other graph learning strategies.

- **Independent Components**: The `ProGNNLearner` and `ELRGNNLearner` classes in `src/models.py` encapsulate all core logic for graph structure learning:
  - **ProGNNLearner**: Low-rank, sparsity, feature smoothness constraints, and PGD optimization
  - **ELRGNNLearner**: Efficient two-stage low-rank estimation (coarse SVD + fine-grained U optimization), hard thresholding, and symmetric normalization
- **Flexible Configuration**: All hyperparameters are managed through YAML configuration files, making experimental settings easy to adjust.
- **Easy Integration**: Compatible with PyTorch Lightning's `GCNWrapper`, enabling both ProGNN and ELR-GCN training modes with a single `graph_learner` parameter. Switch between models by setting `type: prognn` or `type: elrgnn` in configuration files.

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

### Using ProGNN
```bash
# Run Citeseer dataset experiment
python main.py --config config/citeseer.yaml

# Run Cora dataset experiment
python main.py --config config/cora.yaml
```

### Using ELR-GCN

To use ELR-GCN instead of ProGNN, modify your configuration file to set `type: elrgnn` and add ELR-GCN specific parameters:

```yaml
# Example config for ELR-GCN
type: elrgnn  # <--- Set graph learner type

data:
  dataset: cora
  attack: meta
  ptb_rate: 0.15

model:
  hidden: 16
  dropout: 0.5

training:
  lr: 5e-4
  weight_decay: 5e-4
  epochs: 10

# ELR-GCN specific parameters
elrgnn:  # <--- Use 'elrgnn' section instead of 'prognn'
  rank: 50        # Low-rank dimension
  lr_adj: 0.01     # Learning rate for U matrix
  epsilon: 0.01     # Sparsification threshold
  lambda_sim: 1.0   # Similarity loss weight
  lambda_fr: 0.1    # Frobenius norm weight
  inner_steps: 2
  outer_steps: 1
```

## Graph Learner Configuration Guide

This reproduction provides two independent graph learning modules that can be swapped via configuration:

- **`ProGNNLearner`**: Uses PGD optimization with L1 sparsity and nuclear norm constraints
- **`ELRGNNLearner`**: Uses two-stage SVD-based low-rank estimation with hard thresholding

Switch between them by setting `type: prognn` or `type: elrgnn` in your YAML config.

---

## ProGNN Graph Learner

### 1. Instantiating ProGNN Graph Learner

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

---

## ELR-GCN Graph Learner

### 1. Instantiating ELR-GCN Graph Learner

Initializing `ELRGNNLearner` requires only the initial adjacency matrix, configuration parameters, and computing device:

```python
from src.models import ELRGNNLearner

# 1. Prepare initial adjacency matrix (must be Dense format [N, N])
adj_dense = ... 

# 2. Initialize ELR-GCN Learner
# args object needs to include rank, lr_adj, epsilon, lambda_sim and lambda_fr
elr_graph_learner = ELRGNNLearner(adj_dense, args, device)
```

### 2. Integration into GCN Model

Similar to ProGNN, pass the instantiated `elr_graph_learner` to `GCNWrapper`:

```python
from src.model_wrapper import GCNWrapper

model = GCNWrapper(
    nfeat=...,
    nhid=...,
    nclass=...,
    graph_learner=elr_graph_learner,  # <--- Inject ELR-GCN graph learning module
    args=args
)
```

### 3. Key Configuration Parameters for ELR-GCN

| Parameter | Description | Typical Value |
| :--- | :--- | :--- |
| `rank` | **Low-rank rank**: Dimensionality for SVD decomposition (controls computational efficiency) | `50` |
| `lr_adj` | Learning rate for U matrix optimization | `0.01` |
| `epsilon` | **Sparsification threshold**: Hard threshold for edge retention (values ≥ epsilon are kept) | `0.01` |
| `lambda_sim` | **Similarity loss weight**: Controls how closely estimated graph matches original structure | `1.0` |
| `lambda_fr` | **Frobenius norm weight**: Regularizes the component matrix Lambda | `0.1` |
| `outer_steps` | **Structure update steps**: Number of graph structure optimizations per epoch | `1` |
| `inner_steps` | **Parameter update steps**: Number of GCN parameter optimizations per epoch | `2` |

### 4. ELR-GCN Core Mechanism

ELR-GCN uses a two-stage efficient low-rank estimation approach:

1. **Coarse Low-Rank Estimation (Truncated SVD)**: 
   - Performs SVD decomposition: `A ≈ U @ S @ V.T`
   - Fixes singular values `S` (represents high-level graph structure)
   - Only optimizes `U` (singular vector matrix)

2. **Fine-Grained Estimation**:
   - Optimizes `U` using SGD with momentum
   - Reconstructs adjacency: `A_d = U @ S^1/2 @ (U @ S^1/2)^T = U @ S @ U.T`

3. **Sparsification**: Hard thresholding with `epsilon` to remove weak edges

4. **Loss Function**:
   - **Classification Loss (L_CE)**: Standard cross-entropy loss
   - **Similarity Loss (L_Sim)**: `||A - A_d_norm||_F^2` - preserves original structure
   - **Frobenius Loss (L_Fr)**: `||Λ||_F^2` - regularizes component matrix
   - **Total Loss**: `L_CE + λ_sim * L_Sim + λ_fr * L_Fr`

This approach efficiently defends against structural attacks while maintaining computational efficiency through low-rank structure.