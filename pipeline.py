import os
import json
import torch
from torch_geometric.utils import to_dense_adj
from src.config_loader import Config
from src.utils import *
from src.data_loader import PoisonGraphDataModule
from src.model_wrapper import GCNWrapper
import pytorch_lightning as pl
from message import msg

class Pipeline:
    def __init__(self, args: Config):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    def run(self):
        set_seed(self.args.seed)
        
        # 1. Load Original Data (Benchmark)
        # 根据 pipeline.py#L8-9 初始化原始 benchmark
        msg.info(f"[{self.args.dataset}] Loading original benchmark data...")
        data_path = os.path.join('data', f'{self.args.dataset}.npz')
        split_path = os.path.join('splits', f'{self.args.dataset}_prognn_splits.json')
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at: {data_path}")
            
        # Initialize DataModule (with LCC processing as per default)
        self.dm = PoisonGraphDataModule(data_path, split_path if os.path.exists(split_path) else None)
        self.dm.setup()
        
        msg.info(f"Original Graph: {self.dm.data.num_nodes} nodes, {self.dm.data.num_edges} edges")
        
        # Save clean edge_index for later evaluation
        self.clean_edge_index = self.dm.data.edge_index.clone()

        # 2. Inject Attack Data (if configured)
        # 根据 config 中的 attack 类型 (meta or nettack) 修改 data module
        if hasattr(self.args, 'attack') and self.args.attack in ['meta', 'nettack']:
            self._inject_attack_data()
            
        # 3. Ready for Training
        msg.success("Data preparation complete. Ready for model training.")

        # 4. Initialize Model
        msg.info("Initializing ProGNN model...")
        nfeat = self.dm.data.x.shape[1]
        nclass = int(self.dm.data.y.max()) + 1
        
        # === Prepare Graph Learner ===
        # Convert Sparse Edge Index to Dense Adjacency Matrix
        # Note: to_dense_adj returns [Batch, N, N], we need [N, N] for EstimateAdj
        msg.info(f"Converting graph to dense format for {self.args.type}...")
        adj_dense = to_dense_adj(self.dm.data.edge_index, max_num_nodes=self.dm.data.num_nodes)[0]
        adj_dense = adj_dense.to(self.device)
        
        # Initialize Learner strategy
        # Important: pass device to ensure parameters are created on correct device
        if self.args.type == 'elrgnn':
            from src.models import ELRGNNLearner
            msg.info("Initializing ELR-GNN Learner...")
            graph_learner = ELRGNNLearner(adj_dense, self.args, self.device)
        else:
            from src.models import ProGNNLearner
            msg.info("Initializing ProGNN Learner...")
            graph_learner = ProGNNLearner(adj_dense, self.args, self.device)
        
        # Config attributes are flattened, so we access them directly
        # model: hidden, dropout
        # training: lr, weight_decay, epochs
        model = GCNWrapper(
            nfeat=nfeat,
            nhid=self.args.hidden,
            nclass=nclass,
            dropout=self.args.dropout,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            graph_learner=graph_learner,
            args=self.args
        )
        
        # 5. Trainer Setup
        msg.training(f"Starting training for {self.args.epochs} epochs...")
        trainer = pl.Trainer(
            max_epochs=self.args.epochs,
            accelerator='auto',
            devices=1,
            enable_checkpointing=True,
            callbacks=[pl.callbacks.ModelCheckpoint(monitor='val_acc', mode='max')]
        )
        
        # 6. Fit and Test
        trainer.fit(model, datamodule=self.dm)
        
        # Note: We skip trainer.test() here because ProGNN/ELR-GNN relies on a dynamically learned graph structure (best_graph)
        # which is not automatically restored by Lightning's checkpoint mechanism in a new test session.
        # Instead, we perform a comprehensive manual evaluation below using the best checkpoint and reconstructing the graph.
        
        # === 7. Comprehensive Evaluation ===
        msg.info("Starting comprehensive evaluation...")
        
        model.to(self.device)
        model.eval()
        
        eval_metrics = {}
        
        x = self.dm.data.x.to(self.device)
        y = self.dm.data.y.to(self.device)
        test_mask = self.dm.data.test_mask.to(self.device)
        
        msg.info(f"Using best checkpoint at val_acc={float(model.best_val_acc):.6f}")
        
        # Directly load best weights and graph structure
        # If they are None, it indicates a training failure (e.g., accuracy <= 0 or no validation steps), 
        # so letting it raise an AttributeError/TypeError is appropriate.
        model.model.load_state_dict(model.best_weights)
        msg.success("Best model weights loaded.")

        adj_for_gcn = model.best_graph.to(self.device)
        msg.success("Best graph structure loaded.")
        
        learned_adj_raw = adj_for_gcn.clone()
        
        # A. Evaluate Node Classification
        with torch.no_grad():
            logits = model(x, adj_for_gcn)
            class_metrics = evaluate_classification(logits, y, test_mask)
            eval_metrics.update(class_metrics)
            msg.info(f"Node Classification Metrics: {json.dumps(class_metrics, indent=2)}")

        # B. Evaluate Graph Structure Reconstruction
        with torch.no_grad():
            struct_metrics = evaluate_graph_structure(
                learned_adj_raw, 
                self.clean_edge_index.to(self.device), 
                self.dm.data.num_nodes
            )
            eval_metrics.update(struct_metrics)
            msg.info(f"Graph Structure Metrics: {json.dumps(struct_metrics, indent=2)}")
            
        # C. Save Results to Lightning Log
        if trainer.logger and trainer.logger.log_dir:
            log_dir = trainer.logger.log_dir
            save_path = os.path.join(log_dir, 'evaluation_metrics.json')
            with open(save_path, 'w') as f:
                json.dump(eval_metrics, f, indent=4)
            msg.success(f"Evaluation metrics saved to {save_path}")
        else:
            msg.warning("Logger not available, skipping metric save.")

        msg.success("Pipeline execution completed successfully.")
        
    def _inject_attack_data(self):
        """
        Load attacked adjacency matrix and replace the edge_index in DataModule.
        Delegates the heavy lifting to src.utils.load_attacked_data.
        """
        # Call the utility function to load data
        edge_index, target_mask = load_attacked_data(self.args, num_nodes=self.dm.data.num_nodes)
        
        if edge_index is not None:
            old_edges = self.dm.data.num_edges
            self.dm.data.edge_index = edge_index
            new_edges = edge_index.size(1)
            
            # If nettack, store target nodes for evaluation
            if target_mask is not None:
                self.dm.data.target_mask = target_mask
                
            msg.info(f"Attacked Graph injected: {self.dm.data.num_nodes} nodes, {new_edges} edges (was {old_edges})")
