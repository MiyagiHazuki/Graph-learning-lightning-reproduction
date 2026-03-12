import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    """
    Standard GCN implementation using PyTorch Geometric.
    """
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super(GCN, self).__init__()
        
        # PyG GCNConv (default: bias=True)
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nclass)
        
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass.
        
        Args:
            x (Tensor): Node feature matrix [num_nodes, nfeat]
            edge_index (LongTensor): Graph connectivity [2, num_edges]
            edge_weight (Tensor, optional): Edge weights [num_edges]
        """
        # First Graph Convolution Layer
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second Graph Convolution Layer
        x = self.conv2(x, edge_index, edge_weight)
        
        # Log Softmax for classification
        return F.log_softmax(x, dim=1)
