import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from .models import GCN
from copy import deepcopy

class GCNWrapper(pl.LightningModule):
    """
    PyTorch Lightning Wrapper for GCN, supporting both standard training and ProGNN graph structure learning.
    """
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, graph_learner=None, args=None):
        """
        Args:
            nfeat (int): Input feature dimension
            nhid (int): Hidden layer dimension
            nclass (int): Number of classes
            dropout (float): Dropout rate
            lr (float): Learning rate
            weight_decay (float): Weight decay (L2 regularization)
            graph_learner (nn.Module, optional): Module for learning graph structure (e.g., ProGNNLearner).
            args (Namespace, optional): Configuration arguments for ProGNN.
        """
        super().__init__()
        # Ignore graph_learner and args in hparams saving to avoid pickling issues or large logs
        self.save_hyperparameters(ignore=['graph_learner', 'args'])
        
        self.model = GCN(nfeat, nhid, nclass, dropout)
        self.graph_learner = graph_learner
        self.args = args

        # If a graph learner is provided, we switch to manual optimization
        if self.graph_learner is not None:
            self.automatic_optimization = False
            
        # Storage for best graph structure
        self.best_val_acc = 0
        self.best_graph = None
        self.best_weights = None

    def forward(self, x, adj, edge_weight=None):
        # GCN.forward handles both Sparse (edge_index) and Dense (adj)
        return self.model(x, adj, edge_weight)

    def training_step(self, batch, batch_idx):
        if self.graph_learner is None:
            # === Standard GCN Training ===
            out = self(batch.x, batch.edge_index)
            loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
            acc = (out[batch.train_mask].argmax(dim=1) == batch.y[batch.train_mask]).float().mean()
            
            self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
            return loss
        else:
            # === ProGNN Training (Manual Optimization) ===
            opt_gcn = self.optimizers()
            
            # Phase 1: Update Graph Structure (Outer Loop)
            # We delegate the complexity to the graph learner strategy
            for _ in range(self.args.outer_steps):
                self.graph_learner.step(self, self.model, batch.x, batch.y, batch.train_mask)
            
            # Phase 2: Update GCN (Inner Loop)
            for i in range(self.args.inner_steps):
                self.model.train()
                opt_gcn.zero_grad()
                
                # Get current estimated graph (Dense [N, N])
                adj = self.graph_learner()
                
                # Forward pass with Dense Adj
                # GCN.forward will handle unsqueeze internally if needed
                output = self.model(batch.x, adj)
                
                loss_gcn = F.nll_loss(output[batch.train_mask], batch.y[batch.train_mask])
                acc_train = (output[batch.train_mask].argmax(dim=1) == batch.y[batch.train_mask]).float().mean()
                
                self.manual_backward(loss_gcn)
                opt_gcn.step()
                
                # Log only the last inner step
                if i == self.args.inner_steps - 1:
                    self.log('train_loss', loss_gcn, prog_bar=True)
                    self.log('train_acc', acc_train, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        if self.graph_learner is None:
            # Standard validation
            adj = batch.edge_index
        else:
            # ProGNN validation: Use learned graph
            self.graph_learner.eval() # Ensure eval mode for estimator
            adj = self.graph_learner()
        
        out = self(batch.x, adj)
        loss = F.nll_loss(out[batch.val_mask], batch.y[batch.val_mask])
        acc = (out[batch.val_mask].argmax(dim=1) == batch.y[batch.val_mask]).float().mean()
        
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        
        # Save best graph structure for ProGNN
        if self.graph_learner is not None:
            if acc > self.best_val_acc:
                self.best_val_acc = acc
                # CRITICAL FIX: Use .clone() to ensure we save a COPY of the graph, 
                # not a reference to the changing parameter memory.
                self.best_graph = adj.detach().clone()
                self.best_weights = deepcopy(self.model.state_dict())
        
        return loss

    def test_step(self, batch, batch_idx):
        if self.graph_learner is None:
            adj = batch.edge_index
        else:
            # Use best found graph or current graph
            adj = self.best_graph if self.best_graph is not None else self.graph_learner()
            
        out = self(batch.x, adj)
        loss = F.nll_loss(out[batch.test_mask], batch.y[batch.test_mask])
        acc = (out[batch.test_mask].argmax(dim=1) == batch.y[batch.test_mask]).float().mean()
        
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # GCN Optimizer
        # Whether in automatic (GCN only) or manual (ProGNN) mode, 
        # we register the GCN optimizer with Lightning to handle state saving and access.
        return torch.optim.Adam(self.model.parameters(), 
                               lr=self.hparams.lr, 
                               weight_decay=self.hparams.weight_decay)
