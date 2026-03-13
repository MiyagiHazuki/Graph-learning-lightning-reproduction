import random
import os
import numpy as np
import torch
import scipy.sparse as sp
import json
import sys
import traceback
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, classification_report
from torch_geometric.utils import to_dense_adj

# Ensure root directory is in path to import message
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from message import msg

def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

def evaluate_classification(output, labels, mask):
    """
    Evaluate node classification metrics.
    
    Args:
        output: Logits from the model [N, n_class]
        labels: Ground truth labels [N]
        mask: Boolean mask for the set to evaluate (e.g., test_mask)
        
    Returns:
        dict: Dictionary containing Acc, F1 (macro/micro), Precision, Recall
    """
    # Extract masked data and move to CPU
    if mask is not None:
        preds = output[mask].max(1)[1].cpu().numpy()
        y_true = labels[mask].cpu().numpy()
    else:
        preds = output.max(1)[1].cpu().numpy()
        y_true = labels.cpu().numpy()
        
    metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1_macro": float(f1_score(y_true, preds, average='macro')),
        "f1_micro": float(f1_score(y_true, preds, average='micro')),
        "precision_macro": float(precision_score(y_true, preds, average='macro', zero_division=0)),
        "recall_macro": float(recall_score(y_true, preds, average='macro', zero_division=0))
    }
    
    # Optional: Detailed classification report
    # report = classification_report(y_true, preds, output_dict=True)
    
    return metrics

def evaluate_graph_structure(learned_adj, clean_edge_index, num_nodes):
    """
    Evaluate graph structure reconstruction performance.
    
    Args:
        learned_adj: Learned dense adjacency matrix [N, N] (Torch Tensor)
        clean_edge_index: Original clean graph edge_index [2, E] (Torch Tensor)
        num_nodes: Number of nodes in the graph
        
    Returns:
        dict: Dictionary containing AUC, AP, and Frobenius Norm Difference
    """
    msg.info("Evaluating graph structure reconstruction... (This may take a moment for large graphs)")
    
    # Ensure learned_adj is on CPU and detached
    learned_adj = learned_adj.detach().cpu()
    
    # 1. Prepare Clean Adjacency (Ground Truth)
    # Convert clean edge_index to dense [N, N]
    clean_adj = to_dense_adj(clean_edge_index, max_num_nodes=num_nodes)[0].cpu()
    
    # 2. Frobenius Norm Difference
    # ||S - A||_F
    diff_norm = torch.norm(learned_adj - clean_adj, p='fro').item()
    
    # 3. AUC and AP (Treat as Link Prediction)
    # Flatten matrices to vectors
    # Note: For very large graphs, this might consume significant memory.
    # We use numpy for sklearn metrics
    y_true = clean_adj.view(-1).numpy()
    y_score = learned_adj.view(-1).numpy()
    
    try:
        auc = roc_auc_score(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
    except ValueError as e:
        msg.warning(f"Error calculating AUC/AP: {e}")
        auc = 0.0
        ap = 0.0
        
    return {
        "graph_frobenius_norm": float(diff_norm),
        "graph_auc": float(auc),
        "graph_ap": float(ap)
    }

def load_attacked_data(args, num_nodes=None):
    """
    Load attacked adjacency matrix and optional target nodes based on configuration.
    
    Args:
        args: Config object containing 'attack', 'dataset', 'ptb_rate'.
        num_nodes: Optional, for validation checking.
        
    Returns:
        tuple: (edge_index, target_nodes_mask)
            - edge_index: torch.LongTensor of shape [2, E]
            - target_nodes_mask: torch.BoolTensor of shape [N] or None if not applicable
    """
    attack_type = args.attack
    dataset = args.dataset
    ptb_rate = args.ptb_rate
    
    adj_path = None
    target_nodes_path = None
    
    if attack_type == 'meta':
        # Construct path: meta/{dataset}_meta_adj_{rate}.npz
        filename = f"{dataset}_meta_adj_{ptb_rate}.npz"
        adj_path = os.path.join('meta', filename)
        
    elif attack_type == 'nettack':
        # Construct path: nettack/{dataset}_nettack_adj_{perturbations}.npz
        try:
            rate_val = float(ptb_rate)
            filename = f"{dataset}_nettack_adj_{rate_val:.1f}.npz"
        except ValueError:
            filename = f"{dataset}_nettack_adj_{ptb_rate}.npz"
            
        adj_path = os.path.join('nettack', filename)
        target_nodes_path = os.path.join('nettack', f"{dataset}_nettacked_nodes.json")
        
    if not adj_path or not os.path.exists(adj_path):
        msg.warning(f"Warning: Attack file not found at {adj_path}")
        return None, None

    msg.info(f"Loading {attack_type} attacked adjacency matrix from {adj_path}...")
    try:
        # Load the sparse matrix
        adj = sp.load_npz(adj_path)
        
        # Preprocessing for Attack Data
        adj = adj + adj.T
        adj[adj > 1] = 1
        adj.setdiag(0)
        adj.eliminate_zeros()
        
        # Validation
        if num_nodes is not None and adj.shape[0] != num_nodes:
            msg.warning(f"Warning: Attacked adjacency shape {adj.shape} does not match graph nodes {num_nodes}.")
        
        # Convert to PyG Edge Index
        coo = adj.tocoo()
        edge_index = torch.LongTensor([coo.row, coo.col])
        
        # Load target nodes for nettack
        target_mask = None
        if target_nodes_path and os.path.exists(target_nodes_path):
            msg.info(f"Loading target nodes from {target_nodes_path}...")
            with open(target_nodes_path, 'r') as f:
                target_data = json.load(f)
                
                if 'attacked_test_nodes' in target_data:
                    attacked_indices = target_data['attacked_test_nodes']
                    
                    if num_nodes is not None:
                        target_mask = torch.zeros(num_nodes, dtype=torch.bool)
                        # Ensure indices are valid
                        valid_indices = [idx for idx in attacked_indices if idx < num_nodes]
                        if len(valid_indices) != len(attacked_indices):
                            msg.warning("Warning: Some attacked nodes indices are out of bounds or filtered by LCC.")
                        
                        target_mask[valid_indices] = True
                        msg.info(f"Loaded {len(valid_indices)} target nodes for evaluation.")
                
        return edge_index, target_mask

    except Exception as e:
        msg.error(f"Error loading attack data: {e}")
        msg.error(traceback.format_exc())
        return None, None
