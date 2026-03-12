ProGNN 旨在通过探索真实图的内在属性来增强图神经网络对对抗性结构攻击的鲁棒性。数据分 3 个核心目录：

```text
.
├── meta/                   # 全局攻击 (Metattack) 扰动数据
│   ├── {dataset}_meta_adj_{rate}.npz
│   └── ...
├── nettack/                # 针对性攻击 (Nettack) 扰动数据
│   ├── {dataset}_nettack_adj_{perturbations}.npz
│   ├── {dataset}_nettacked_nodes.json
│   └── ...
└── splits/                 # 数据集划分 (Train/Val/Test)
    ├── {dataset}_prognn_splits.json
    └── ...
```

**meta** ：存放不同全局扰动率下 Metattack 攻击的扰动图结构，文件格式 `{dataset}_meta_adj_{rate}.npz` ， `{rate}` 为全局扰动率（被修改边占总边数比例），用于测试全局结构破坏下的鲁棒性。

**nettack** ：存放针对性攻击数据，含 `{dataset}_nettack_adj_{perturbations}.npz` （受损邻接矩阵， `{perturbations}` 为单目标节点边修改次数）、 `{dataset}_nettacked_nodes.json` （被攻击节点索引），用于评估局部节点攻击下的分类稳定性。

**splits** ：存放 `{dataset}_prognn_splits.json` （固定训练 / 验证 / 测试集划分索引），避免随机划分导致的性能评估偏差，保障模型对比公平性。