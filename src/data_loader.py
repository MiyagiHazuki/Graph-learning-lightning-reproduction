import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
import json
import os
import sys

# Ensure root directory is in path to import message
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from message import msg

class PoisonGraphDataModule(pl.LightningDataModule):
    def __init__(self, data_path, split_path=None, batch_size=1, require_lcc=True):
        """
        Args:
            data_path: Path to the .npz file (e.g., 'data/cora.npz')
            split_path: Path to the split .json file (e.g., 'splits/cora_prognn_splits.json')
            batch_size: Batch size for DataLoader (default: 1 for full-batch training)
            require_lcc: Whether to select the largest connected component (default: True)
        """
        super().__init__()
        self.data_path = data_path
        self.split_path = split_path
        self.batch_size = batch_size
        self.require_lcc = require_lcc
        self.data = None

    def setup(self, stage=None):
        if self.data is not None:
            return

        msg.info(f"Loading data from {self.data_path}...")
        # 1. Load NPZ file
        raw = np.load(self.data_path, allow_pickle=True)
        
        # 2. Reconstruct sparse matrices
        adj = sp.csr_matrix((raw['adj_data'], raw['adj_indices'], raw['adj_indptr']), 
                           shape=raw['adj_shape'])
        feat = sp.csr_matrix((raw['attr_data'], raw['attr_indices'], raw['attr_indptr']), 
                            shape=raw['attr_shape'])
        labels = raw['labels']

        # === Preprocessing Steps (Aligned with deep_robust.py) ===
        
        # Step A: Symmetrize
        adj = adj + adj.T
        adj = adj.tolil()
        
        # Step B: Binarize (unweighted)
        adj[adj > 1] = 1

        # Step C: Select Largest Connected Component (LCC)
        if self.require_lcc:
            msg.info("Selecting Largest Connected Component (LCC)...")
            _, component_indices = sp.csgraph.connected_components(adj)
            component_sizes = np.bincount(component_indices)
            # Find the component with the largest size
            largest_component_idx = np.argmax(component_sizes)
            # Get nodes belonging to the largest component
            nodes_to_keep = np.where(component_indices == largest_component_idx)[0]
            
            msg.info(f"Original nodes: {adj.shape[0]}, Nodes after LCC: {len(nodes_to_keep)}")
            
            # Filter adj, features, and labels
            adj = adj[nodes_to_keep][:, nodes_to_keep]
            feat = feat[nodes_to_keep]
            labels = labels[nodes_to_keep]
            
            # We need to map old indices to new indices to handle splits correctly
            # old_idx -> new_idx (if kept), else -1
            old_to_new_map = np.full(raw['adj_shape'][0], -1, dtype=int)
            old_to_new_map[nodes_to_keep] = np.arange(len(nodes_to_keep))

        # Step D: Remove Self-loops
        adj.setdiag(0)
        adj = adj.astype("float32").tocsr()
        adj.eliminate_zeros()

        # Final checks
        assert np.abs(adj - adj.T).sum() == 0, "Graph is not symmetric"
        assert adj.max() == 1, "Graph must be unweighted"

        # === Convert to PyTorch ===
        
        # Features: Dense tensor (N x F)
        x = torch.FloatTensor(feat.todense())
        
        # Labels: Long tensor (N)
        y = torch.LongTensor(labels)
        
        # Adjacency: COO format for PyG (2 x E)
        coo = adj.tocoo()
        edge_index = torch.LongTensor([coo.row, coo.col])

        # Create PyG Data object
        self.data = Data(x=x, edge_index=edge_index, y=y)
        
        # === Handle Splits ===
        if self.split_path and os.path.exists(self.split_path):
            msg.info(f"Loading splits from {self.split_path}...")
            with open(self.split_path, 'r') as f:
                splits = json.load(f)
            
            num_nodes = x.size(0)
            
            # Helper to create boolean masks
            def create_mask(indices):
                mask = torch.zeros(num_nodes, dtype=torch.bool)
                # The split indices are already based on the LCC graph, so we use them directly.
                mask[indices] = True
                return mask

            self.data.train_mask = create_mask(splits['idx_train'])
            self.data.val_mask = create_mask(splits['idx_val'])
            self.data.test_mask = create_mask(splits['idx_test'])
        else:
            msg.warning("Warning: No split path provided or file not found. Masks not set.")

    def train_dataloader(self):
        return DataLoader([self.data], batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader([self.data], batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader([self.data], batch_size=self.batch_size)
