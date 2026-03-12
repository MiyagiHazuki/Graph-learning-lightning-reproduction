import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from .models import GCN

class GCNWrapper(pl.LightningModule):
    """
    PyTorch Lightning Wrapper for GCN.
    """
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4):
        """
        Args:
            nfeat (int): Input feature dimension
            nhid (int): Hidden layer dimension
            nclass (int): Number of classes
            dropout (float): Dropout rate
            lr (float): Learning rate
            weight_decay (float): Weight decay (L2 regularization)
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.model = GCN(nfeat, nhid, nclass, dropout)

    def forward(self, x, edge_index, edge_weight=None):
        return self.model(x, edge_index, edge_weight)

    def training_step(self, batch, batch_idx):
        """
        Standard training step for node classification.
        """
        out = self(batch.x, batch.edge_index)
        
        # Only compute loss on training nodes
        loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
        
        # Calculate accuracy
        acc = (out[batch.train_mask].argmax(dim=1) == batch.y[batch.train_mask]).float().mean()
        
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Standard validation step.
        """
        out = self(batch.x, batch.edge_index)
        
        # Only compute loss on validation nodes
        loss = F.nll_loss(out[batch.val_mask], batch.y[batch.val_mask])
        acc = (out[batch.val_mask].argmax(dim=1) == batch.y[batch.val_mask]).float().mean()
        
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        """
        Standard test step.
        """
        out = self(batch.x, batch.edge_index)
        
        loss = F.nll_loss(out[batch.test_mask], batch.y[batch.test_mask])
        acc = (out[batch.test_mask].argmax(dim=1) == batch.y[batch.test_mask]).float().mean()
        
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        """
        Configure optimizer.
        """
        return torch.optim.Adam(self.parameters(), 
                               lr=self.hparams.lr, 
                               weight_decay=self.hparams.weight_decay)
