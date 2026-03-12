import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, DenseGCNConv
from deeprobust.graph.defense.pgd import PGD, prox_operators
import torch.optim as optim

class GCN(nn.Module):
    """
    Standard GCN implementation using PyTorch Geometric, supporting both Sparse and Dense inputs.
    """
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super(GCN, self).__init__()
        
        # PyG GCNConv (default: bias=True)
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nclass)
        
        # PyG DenseGCNConv for ProGNN
        self.dense_conv1 = DenseGCNConv(nfeat, nhid)
        self.dense_conv2 = DenseGCNConv(nhid, nclass)
        
        self.dropout = dropout

    def forward(self, x, adj, edge_weight=None):
        """
        Forward pass supporting both Sparse (edge_index) and Dense (adj matrix).
        
        Args:
            x (Tensor): Node feature matrix [num_nodes, nfeat] or [batch, num_nodes, nfeat]
            adj (Tensor): 
                - If LongTensor [2, E]: Treated as edge_index (Sparse)
                - If FloatTensor [N, N] or [B, N, N]: Treated as adjacency matrix (Dense)
            edge_weight (Tensor, optional): Edge weights [num_edges] (Only for Sparse)
        """
        is_dense = False
        if isinstance(adj, torch.Tensor) and adj.is_floating_point():
            is_dense = True
            if adj.dim() == 2:
                adj = adj.unsqueeze(0) # [N, N] -> [1, N, N]
            if x.dim() == 2:
                x = x.unsqueeze(0) # [N, F] -> [1, N, F]

        if is_dense:
            # Dense Path
            x = self.dense_conv1(x, adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.dense_conv2(x, adj)
            
            # Remove batch dimension if input was not batched
            if x.size(0) == 1:
                x = x.squeeze(0)
        else:
            # Sparse Path (Standard PyG)
            edge_index = adj # Rename for clarity
            x = self.conv1(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index, edge_weight)
        
        # Log Softmax for classification
        return F.log_softmax(x, dim=1)


class EstimateAdj(nn.Module):
    """Provide a pytorch parameter matrix for estimated
    adjacency matrix and corresponding operations.
    """

    def __init__(self, adj, symmetric=False, device='cpu'):
        super(EstimateAdj, self).__init__()
        n = len(adj)
        self.estimated_adj = nn.Parameter(torch.FloatTensor(n, n).to(device))
        self._init_estimation(adj)
        self.symmetric = symmetric
        self.device = device

    def _init_estimation(self, adj):
        with torch.no_grad():
            n = len(adj)
            self.estimated_adj.data.copy_(adj)

    def forward(self):
        return self.estimated_adj

    def normalize(self):

        if self.symmetric:
            adj = (self.estimated_adj + self.estimated_adj.t())/2
        else:
            adj = self.estimated_adj

        normalized_adj = self._normalize(adj + torch.eye(adj.shape[0]).to(self.device))
        return normalized_adj

    def _normalize(self, mx):
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx


class ProGNNLearner(nn.Module):
    """
    Encapsulates the graph structure learning logic of ProGNN.
    Responsible for updating the adjacency matrix using PGD and various losses.
    """
    def __init__(self, adj, args, device):
        super().__init__()
        self.estimator = EstimateAdj(adj, symmetric=args.symmetric, device=device)
        self.args = args
        self.device = device
        
        # Initialize optimizers for graph structure
        self.optimizer_adj = optim.SGD(self.estimator.parameters(),
                              momentum=0.9, lr=args.lr_adj)

        self.optimizer_l1 = PGD(self.estimator.parameters(),
                        proxs=[prox_operators.prox_l1],
                        lr=args.lr_adj, alphas=[args.alpha])

        self.optimizer_nuclear = PGD(self.estimator.parameters(),
                  proxs=[prox_operators.prox_nuclear_cuda],
                  lr=args.lr_adj, alphas=[args.beta])
        
        # Save original adj for loss calculation
        # Use register_buffer so it moves to device automatically but isn't a parameter
        self.register_buffer('adj_orig', adj)

    def forward(self):
        """Returns the normalized adjacency matrix for GCN training."""
        return self.estimator.normalize()

    def step(self, pl_module, gcn_model, features, labels, idx_train):
        """
        Performs one step of graph structure optimization (Outer Loop).
        """
        args = self.args
        estimator = self.estimator
        
        self.estimator.train()
        self.optimizer_adj.zero_grad()

        loss_l1 = torch.norm(estimator.estimated_adj, 1)
        loss_fro = torch.norm(estimator.estimated_adj - self.adj_orig, p='fro')
        normalized_adj = estimator.normalize()

        if args.lambda_:
            loss_smooth_feat = self.feature_smoothing(estimator.estimated_adj, features)
        else:
            loss_smooth_feat = 0 * loss_l1

        # Use GCN to get prediction on current graph
        # Note: GCN expects [Batch, N, N] for DenseGCNConv
        # We manually handle unsqueeze inside GCN.forward if needed, 
        # but here we pass normalized_adj [N, N] which GCN.forward handles.
        output = gcn_model(features, normalized_adj)
        loss_gcn = F.nll_loss(output[idx_train], labels[idx_train])
        
        loss_symmetric = torch.norm(estimator.estimated_adj \
                        - estimator.estimated_adj.t(), p="fro")

        loss_diffiential =  loss_fro + args.gamma * loss_gcn + args.lambda_ * loss_smooth_feat + args.phi * loss_symmetric

        # Manual backward via Lightning Module to handle precision/plugins correctly if needed
        # But since we are in manual optimization mode, we can call backward directly 
        # or use pl_module.manual_backward(loss) if we want Lightning hooks.
        pl_module.manual_backward(loss_diffiential)

        self.optimizer_adj.step()
        
        # Proximal updates (PGD)
        if args.beta != 0:
            self.optimizer_nuclear.zero_grad()
            self.optimizer_nuclear.step()

        self.optimizer_l1.zero_grad()
        self.optimizer_l1.step()

        # Projection to [0, 1]
        estimator.estimated_adj.data.copy_(torch.clamp(
                  estimator.estimated_adj.data, min=0, max=1))
        
        return loss_diffiential.item()

    def feature_smoothing(self, adj, X):
        adj = (adj.t() + adj)/2
        rowsum = adj.sum(1)
        r_inv = rowsum.flatten()
        D = torch.diag(r_inv)
        L = D - adj

        r_inv = r_inv  + 1e-3
        r_inv = r_inv.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        # L = r_mat_inv @ L
        L = r_mat_inv @ L @ r_mat_inv

        XLXT = torch.matmul(torch.matmul(X.t(), L), X)
        loss_smooth_feat = torch.trace(XLXT)
        return loss_smooth_feat
