import random
import os
import numpy as np
import torch
import scipy.sparse as sp
import json

def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

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
        print(f"Warning: Attack file not found at {adj_path}")
        return None, None

    print(f"Loading {attack_type} attacked adjacency matrix from {adj_path}...")
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
            print(f"Warning: Attacked adjacency shape {adj.shape} does not match graph nodes {num_nodes}.")
        
        # Convert to PyG Edge Index
        coo = adj.tocoo()
        edge_index = torch.LongTensor([coo.row, coo.col])
        
        # Load target nodes for nettack
        target_mask = None
        if target_nodes_path and os.path.exists(target_nodes_path):
            print(f"Loading target nodes from {target_nodes_path}...")
            with open(target_nodes_path, 'r') as f:
                target_data = json.load(f)
                
                if 'attacked_test_nodes' in target_data:
                    attacked_indices = target_data['attacked_test_nodes']
                    
                    if num_nodes is not None:
                        target_mask = torch.zeros(num_nodes, dtype=torch.bool)
                        # Ensure indices are valid
                        valid_indices = [idx for idx in attacked_indices if idx < num_nodes]
                        if len(valid_indices) != len(attacked_indices):
                            print(f"Warning: Some attacked nodes indices are out of bounds or filtered by LCC.")
                        
                        target_mask[valid_indices] = True
                        print(f"Loaded {len(valid_indices)} target nodes for evaluation.")
                
        return edge_index, target_mask

    except Exception as e:
        print(f"Error loading attack data: {e}")
        import traceback
        traceback.print_exc()
        return None, None
