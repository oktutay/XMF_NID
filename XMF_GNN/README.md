# XMF-GNN

PyTorch / PyG implementation of:

> Ma Z., Liu Y., Chen Y., Liu Z., Li Y. **XMF-GNN: A cross-modality dynamic
> fusion heterogeneous graph neural network for network intrusion detection.**
> *Neurocomputing* 655 (2025) 131285.
> [DOI: 10.1016/j.neucom.2025.131285](https://doi.org/10.1016/j.neucom.2025.131285)

The authors did not publish reference code (paper says *"Data will be made
available on request"*). This repository builds a paper-faithful
implementation by:

1. Reusing the proven NFStream-based flow/packet feature extractor and the
   heterogeneous graph dataset class from [GNN4ID](https://github.com/Yasir-ali-farrukh/GNN4ID)
   (Farrukh et al., *Expert Systems with Applications* 2025), which the
   XMF-GNN paper itself credits as the graph construction baseline (Sec. 2.2
   and Sec. 3, "the heterogeneous graph construction method proposed by
   Farrukh et al.").
2. Re-implementing the components that XMF-GNN modifies relative to XG-NID:
   - **Multi-scale temporal feature extractor** (paper Sec. 3.1 / Algorithm 1):
     four sliding windows {10 s, 30 s, 60 s, 300 s} × 13 features per window
     → 52 multi-scale features stacked with the 76 base flow features
     ([`Utility/Additional_Features.py`](Utility/Additional_Features.py)).
   - **Edge-feature folding** (paper Sec. 3.2): the `contain` and `link` edge
     types and their attributes are removed and folded into the corresponding
     packet-node attributes; only a single `connected_to` relation remains
     ([`Utility/Functions.py`](Utility/Functions.py)).
   - **Two-layer SAGE-based HGNN encoder** (paper Sec. 3.3, eq. 4-7):
     `SAGEConv → BN → LeakyReLU` × 2 with mean aggregation
     ([`Utility/Model.py`](Utility/Model.py)).
   - **Modality attention fusion module** (paper Sec. 3.4 / Algorithm 2 /
     eq. 10-12): linear projection to a shared d′ space, shared scorer
     `softmax(W_a · tanh(Z))`, weighted combination of the *original*
     embeddings ([`Utility/Model.py`](Utility/Model.py)).
   - **Ablation variants** B-GNN / FM-GNN / SA-GNN / GA-GNN / MA-GNN
     (paper Table 8) — selected via the `fusion=` argument of `XMFGNN`.
   - **Integrated Gradients explainer** (paper Sec. 4.6.4 / eq. 19) —
     reproduces the feature-importance bar charts of Fig. 6 / Fig. 7
     ([`Utility/IG_Explainer.py`](Utility/IG_Explainer.py)).

## Repository layout

```
XMF_GNN/
├── Utility/
│   ├── __init__.py
│   ├── Model.py                              # XMF-GNN architecture + ablations
│   ├── Functions.py                          # NIDSDataset + CIC-IoT2023 utils
│   ├── Additional_Features.py                # multi-scale temporal extractor
│   ├── Feature_extractor_flow_packet_combined.py   # NFStream pipeline
│   ├── Training.py                           # train/test loops + schedulers
│   └── IG_Explainer.py                       # Integrated Gradients
├── XMF_GNN.ipynb                             # build graph dataset from CSV
├── XMF_GNN_Model.ipynb                       # train + evaluate + ablations
├── Data_preprocessing_CIC-IoT2023.ipynb      # paper Sec. 4.2 preprocessing
├── BAOCAO_DOI_CHIEU_XMF_GNN.md               # detailed paper ↔ code report
├── requirements.txt
└── README.md
```

## Architecture (paper Fig. 1)

```
        Raw PCAP
          │
          ▼
[1] NFStream + 76 base flow + 14 packet + 1500-byte payload    Sec. 3.1
          │
          ▼
[2] Multi-scale temporal features ({10,30,60,300}s × 13 = 52)  Sec. 3.1, Alg. 1
          │
          ▼
[3] Heterogeneous graph                                        Sec. 3.2
        flow node feat       = 76 + 52 = 128 dims
        packet node feat     = 14 + 1500 = 1514 dims
        single edge type:  flow ─connected_to─► packet
          │
          ▼
[4] Two-layer SAGE encoder   SAGE → BN → LeakyReLU × 2          Sec. 3.3
          │
          ▼
[5] Per-modality global mean pool     →     h_f, h_p             eq. 8-9
          │
          ▼
[6] ModalityFusion (attention)                                  Sec. 3.4
        z = W h + b              (proj into d' = attn_size)
        α = softmax(W_a · tanh([z_f; z_p]))     ← shared scorer
        h_ζ = α_f · h_f + α_p · h_p
          │
          ▼
[7] Classifier head: LogSoftmax(W₂ · ReLU(W₁ · ReLU(W₀ · h_ζ))) eq. 13
```

## Hyperparameters (paper Table 3)

| Hyperparameter   | Value          |
|------------------|----------------|
| optimizer        | Adam           |
| loss             | NLL            |
| batch size       | 64             |
| learning rate    | [1e-2, 1e-5]   |
| epochs           | 100            |
| hidden size      | 64             |
| attn size        | 32             |
| pooling          | Mean           |
| LR scheduler     | ReduceLROnPlateau (patience=5, threshold=0.01) |

These defaults are wired into `Utility/Training.make_optimizer_and_scheduler`.

## Quickstart

### 1. Install

```bash
conda create -n xmfgnn python=3.10 -y
conda activate xmfgnn
pip install -r requirements.txt
```

NFStream needs `libpcap` on Linux/macOS; on Windows it is easiest to run
the feature extractor inside WSL2.

### 2. From PCAP to flow CSV

```bash
python Utility/Feature_extractor_flow_packet_combined.py path/to/in.pcap path/to/out_dir/
```

### 3. Add multi-scale temporal block

```python
from Utility.Additional_Features import additional_features
additional_features("path/to/in.csv")          # adds 52 features in place
```

### 4. Build the heterogeneous graph dataset

See `XMF_GNN.ipynb` for the canonical notebook flow.

### 5. Train

```python
import torch
from torch_geometric.loader import DataLoader
from Utility import NIDSDataset, XMFGNN, train, test_cm

train_set = NIDSDataset(root='dataset_root', label_dict=..., filename=[...],
                        single_file=True)
test_set  = NIDSDataset(root='dataset_root', label_dict=..., filename=[...],
                        single_file=True, test=True)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=64)

sample = train_set[0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = XMFGNN(
    hetero_metadata=sample.metadata(),
    hidden_size=64, attn_size=32, num_classes=8, fusion='attn',
).to(device)

train(train_loader, model, args={'epochs': 100, 'lr': 1e-2}, device=device)
test_cm(test_loader, model, device=device)
```

### 6. Ablations (paper Table 8)

| Variant   | Argument              |
|-----------|-----------------------|
| B-GNN     | `fusion='baseline'`   |
| FM-GNN    | `fusion='mlp'`        |
| SA-GNN    | `fusion='simple_attn'`|
| GA-GNN    | `fusion='gated'`      |
| MA-GNN    | `fusion='multi_head'` |
| XMF-GNN   | `fusion='attn'`       |

### 7. Integrated Gradient explainer (paper Sec. 4.6.4)

```python
from Utility.IG_Explainer import IntegratedGradientExplainer, top_flow_features

ig = IntegratedGradientExplainer(model, device=device, n_steps=50)
attr = ig.explain(test_set[0])
for name, a, v in top_flow_features(attr, FLOW_FEATURE_NAMES, top_n=10):
    print(f"{name:32s}  attr={a:+.4f}  value={v:+.4f}")
```

## Expected results (paper Sec. 4.6.1)

| Dataset        | Task        | Accuracy | F1 (weighted) |
|----------------|-------------|----------|---------------|
| CIC-IDS2017    | multi-class | 0.974    | 0.977         |
| CIC-IDS2017    | binary      | 0.987    | 0.988         |
| CIC-IoT2023    | multi-class | 0.984    | 0.985         |
| CIC-IoT2023    | binary      | 0.998    | 0.998         |

Per-class recall/F1 numbers are reported in paper Table 6.

## Status

This repository is faithful to the paper text but **has not yet been
benchmarked end-to-end on the real datasets**. See
[`BAOCAO_DOI_CHIEU_XMF_GNN.md`](BAOCAO_DOI_CHIEU_XMF_GNN.md) for a section-by-
section paper ↔ code mapping and a list of design decisions made where the
paper leaves a degree of freedom (e.g. which 13 per-window features make
up the 52-dim multi-scale block).
