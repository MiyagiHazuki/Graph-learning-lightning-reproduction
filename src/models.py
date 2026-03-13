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
            # Disable add_loop because input adj is already normalized with self-loops
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

    def __init__(self, adj, symmetric=False, device='cuda'):
        super(EstimateAdj, self).__init__()
        n = len(adj)
        self.estimated_adj = nn.Parameter(torch.FloatTensor(n, n).to(device))
        self._init_estimation(adj)
        self.symmetric = symmetric
        self.device = device

    def _init_estimation(self, adj):
        with torch.no_grad():
            self.estimated_adj.data.copy_(adj)

    def forward(self):
        return self.estimated_adj

    def symmetrize(self):
        if self.symmetric:
            return (self.estimated_adj + self.estimated_adj.t()) / 2
        return self.estimated_adj

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
        return self.estimator.symmetrize()

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
        adj = estimator.symmetrize()

        if args.lambda_:
            loss_smooth_feat = self.feature_smoothing(estimator.estimated_adj, features)
        else:
            loss_smooth_feat = 0 * loss_l1

        # Use GCN to get prediction on current graph
        # Note: GCN expects [Batch, N, N] for DenseGCNConv
        # We manually handle unsqueeze inside GCN.forward if needed, 
        # but here we pass normalized_adj [N, N] which GCN.forward handles.
        output = gcn_model(features, adj)
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

class ELRGNNLearner(nn.Module):
    """
    ELR-GNN: Efficient Low-Rank Graph Neural Network Defense Against Structural Attacks.
    Paper: https://arxiv.org/abs/2309.10136
    
    Key Mechanism:
    1. Coarse Low-Rank Estimation: Initialize using Truncated SVD.
    2. Fine-Grained Estimation: Optimize U while keeping S fixed.
    3. Sparsification: Hard thresholding (epsilon).
    4. Normalization: Standard GCN normalization.
    """
    def __init__(self, adj, args, device):
        super().__init__()
        self.args = args
        self.device = device
        
        # --- 1. Coarse Low-Rank Estimation (Truncated SVD) ---
        # Get rank from args, default to 50 if not specified
        self.rank = getattr(args, 'rank', 50)
        
        # Perform SVD on the input adjacency matrix
        # Note: torch.svd_lowrank is efficient for large matrices on GPU
        # A ~= U @ S @ V.T. Since A is symmetric, U should be approx V.
        # We use U and S.
        with torch.no_grad():
            U, S, _ = torch.svd_lowrank(adj, q=self.rank)
        
        # S is a vector of singular values. 
        # According to paper, we fix S (singular values) as they represent high-level structure
        # and attacks mostly affect low-rank components (which we filtered out) 
        # or we want to preserve the valid high-rank info.
        # Paper says: "keeping the obtained singular values fixed"
        # Fix: Add abs() and epsilon for numerical stability to avoid NaN in sqrt
        self.register_buffer('S_sqrt', torch.diag(torch.sqrt(torch.abs(S) + 1e-12)))
        
        # U is the singular vector matrix to be optimized
        # Initialize U_d with the SVD result
        self.U = nn.Parameter(U)
        
        # Save original adjacency for Similarity Loss calculation
        # We save it as a dense tensor if it fits in memory, or use sparse if needed.
        # Assuming adj is passed as dense tensor based on pipeline.py
        self.register_buffer('adj_orig', adj)

        # --- 2. Optimizer ---
        # Paper IV-A2: SGD optimizer for learning U_d with momentum 0.9
        self.optimizer_u = optim.SGD([self.U], momentum=0.9, lr=args.lr_adj)

    def _normalize(self, adj):
        """
        Symmetric Normalization: D^-1/2 * A * D^-1/2
        Paper Eq (6): No self-loops added
        """
        # 1. Compute Degree
        degree = adj.sum(1)
        
        # 2. Compute D^-1/2 (with numerical stability)
        d_inv_sqrt = torch.pow(degree + 1e-6, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        
        # 3. Normalize
        return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

    def forward(self):
        """
        Returns the normalized, sparse, low-rank estimated adjacency matrix.
        Process: U, S -> A_d -> Sparse A_d -> Normalized A_d
        """
        # Reconstruction: A_d = (U S^1/2) (U S^1/2)^T = U S U^T
        Lambda = self.U @ self.S_sqrt
        A_d = Lambda @ Lambda.t()
        
        # Sparsification: Hard Thresholding
        # Paper Eq (5): A_d(i,j) = A_d(i,j) if >= epsilon else 0
        # This implicitly filters out negative values since epsilon > 0
        epsilon = getattr(self.args, 'epsilon', 0.01)
        
        # We use a mask for sparsification to maintain gradient flow where values are kept
        mask = (A_d >= epsilon).float()
        A_d_sparse = A_d * mask
        
        return A_d_sparse

    def step(self, pl_module, gcn_model, features, labels, idx_train):
        """
        Performs one step of graph structure optimization (Fine-Grained Estimation).
        """
        self.train()
        self.optimizer_u.zero_grad()
        
        # --- 1. Forward Pass to get Estimated Graph ---
        # Reconstruct
        Lambda = self.U @ self.S_sqrt
        A_d = Lambda @ Lambda.t()
        
        # Sparsify
        epsilon = getattr(self.args, 'epsilon', 0.01)
        # Detach mask to treat it as a fixed structure selection for this step
        mask = (A_d >= epsilon).float().detach()
        A_d_sparse = A_d * mask
        
        # Use A_d_sparse directly (DenseGCNConv handles normalization)
        gcn_adj = A_d_sparse

        # --- 2. GCN Loss (L_CE) ---
        # Pass estimated graph to GCN
        output = gcn_model(features, gcn_adj)
        loss_gcn = F.nll_loss(output[idx_train], labels[idx_train])
        
        # --- 3. ELR-GNN Specific Losses ---
        # Get hyperparameters
        lambda_sim = getattr(self.args, 'lambda_sim', 1.0) # Similarity regularization weight
        lambda_fr = getattr(self.args, 'lambda_fr', 0.1)   # Frobenius regularization weight
        
        # L_Sim: Similarity to original graph (Eq 7)
        # Compare NORMALIZED estimated graph with ORIGINAL RAW graph (Eq 6)
        # Use sum of squares instead of norm**2 to avoid NaN gradients when diff is 0
        gcn_adj_norm = self._normalize(gcn_adj)
        loss_sim = torch.sum((self.adj_orig - gcn_adj_norm) ** 2)
        
        # L_Fr: Frobenius norm of component matrix Lambda (Eq 8)
        loss_fr = torch.norm(Lambda, p='fro') ** 2
        
        # Total Loss (Eq 9)
        loss_total = loss_gcn + lambda_sim * loss_sim + lambda_fr * loss_fr

        # --- 4. Optimization ---
        # Manual backward
        pl_module.manual_backward(loss_total)
        self.optimizer_u.step()
        
        return loss_total.item()
