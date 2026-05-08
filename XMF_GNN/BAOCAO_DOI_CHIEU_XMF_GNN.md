# Báo cáo đối chiếu code XMF-GNN với bài báo gốc

**Mục đích.** Tài liệu này tổng hợp toàn bộ kết quả đối chiếu code repository
[XMF_GNN](.) với bài báo gốc:

> Ma Z., Liu Y., Chen Y., Liu Z., Li Y. **XMF-GNN: A cross-modality dynamic
> fusion heterogeneous graph neural network for network intrusion detection.**
> *Neurocomputing* 655 (2025) 131285.

Tác giả không công bố mã nguồn (mục *Data availability* chỉ ghi *"Data will
be made available on request"*), nên repository này được build dựa trên:

1. **Bộ extractor + dataset class của GNN4ID** (Farrukh et al. 2025 — bài
   XG-NID), được chính paper XMF-GNN khẳng định là baseline graph
   construction (Sec. 2.2 và Sec. 3, *"the heterogeneous graph construction
   method proposed by Farrukh et al."*).
2. **Tái hiện trực tiếp từ paper XMF-GNN** các điểm khác biệt: multi-scale
   temporal extractor (Algorithm 1), edge-feature folding (Sec. 3.2), HGNN
   SAGE encoder (Sec. 3.3), ModalityFusion attention (Sec. 3.4 / Algorithm
   2), 5 ablation variants (Sec. 4.6.3 / Table 8), Integrated Gradient
   explainer (Sec. 4.6.4 / eq. 19).

**Ngày phân tích:** 2026-05-08.

---

## Mục lục

1. [Tóm tắt nhanh](#1-tóm-tắt-nhanh)
2. [Cấu trúc các components của paper](#2-cấu-trúc-các-components-của-paper)
3. [Khác biệt XMF-GNN vs XG-NID](#3-khác-biệt-xmf-gnn-vs-xg-nid)
4. [Chi tiết từng component](#4-chi-tiết-từng-component)
5. [Danh sách file đã tạo](#5-danh-sách-file-đã-tạo)
6. [Cách chạy lại](#6-cách-chạy-lại)
7. [Phụ lục: bảng tham chiếu paper ↔ code](#7-phụ-lục-bảng-tham-chiếu-paper--code)
8. [Các điểm paper không cố định và quyết định thiết kế](#8-các-điểm-paper-không-cố-định-và-quyết-định-thiết-kế)

---

## 1. Tóm tắt nhanh

| # | Component | Vị trí trong paper | File implement |
|---|---|---|---|
| 1 | Flow & Packet Generator (NFStream) | Sec. 3.1 | [Utility/Feature_extractor_flow_packet_combined.py](Utility/Feature_extractor_flow_packet_combined.py) |
| 2 | Multi-scale Temporal Feature Extractor | Sec. 3.1 / Algorithm 1 | [Utility/Additional_Features.py](Utility/Additional_Features.py) |
| 3 | Heterogeneous Graph Construction | Sec. 3.2 | [Utility/Functions.py](Utility/Functions.py) `NIDSDataset` |
| 4 | HGNN SAGE Encoder (2 layer) | Sec. 3.3 / eq. 4-7 | [Utility/Model.py](Utility/Model.py) `HeteroSAGEEncoder` |
| 5 | Modality Attention Fusion | Sec. 3.4 / Algorithm 2 / eq. 10-12 | [Utility/Model.py](Utility/Model.py) `ModalityFusion` |
| 6 | Classifier head + LogSoftmax | eq. 13 | [Utility/Model.py](Utility/Model.py) `XMFGNN` |
| 7 | Ablation variants B/FM/SA/GA/MA-GNN | Sec. 4.6.3 / Table 8 | [Utility/Model.py](Utility/Model.py) (`fusion=` arg) |
| 8 | Training pipeline | Sec. 4.4 / Table 3 | [Utility/Training.py](Utility/Training.py) |
| 9 | Integrated Gradient explainer | Sec. 4.6.4 / eq. 19 | [Utility/IG_Explainer.py](Utility/IG_Explainer.py) |
| 10 | Data preprocessing (MAC filter, balance) | Sec. 4.2 / Table 2-3 | [Utility/Functions.py](Utility/Functions.py) `split_csv`, `Combining_classes` |

---

## 2. Cấu trúc các components của paper

```
Raw PCAP
  │
  ▼
[1] NFStream + 76 base flow + 14 packet + 1500-byte payload    Sec. 3.1
  │
  ▼
[2] Multi-scale Temporal Block ({10s,30s,60s,300s}×13 = 52)    Sec. 3.1, Alg. 1
  │
  ▼
[3] Heterogeneous Graph                                        Sec. 3.2
        flow node feat       = 76 + 52 = 128 dims
        packet node feat     = 14 + 1500 = 1514 dims
        single edge type:  flow ─connected_to─► packet
  │
  ▼
[4] HGNN encoder: SAGE → BN → LeakyReLU × 2                    Sec. 3.3
  │
  ▼
[5] Per-modality global mean pool   →   h_f, h_p                eq. 8-9
  │
  ▼
[6] ModalityFusion                                              Sec. 3.4
        z = W h + b              (proj into d' = attn_size)
        α = softmax(W_a · tanh([z_f; z_p]))     ← shared scorer
        h_ζ = α_f · h_f + α_p · h_p
  │
  ▼
[7] Classifier head: LogSoftmax(W₂ · ReLU(W₁ · ReLU(W₀ · h_ζ))) eq. 13
```

---

## 3. Khác biệt XMF-GNN vs XG-NID

XMF-GNN xây dựng dựa trên kiến trúc của XG-NID (Farrukh et al. 2025) nhưng
**cải tiến 4 điểm cốt lõi**:

| Khía cạnh | XG-NID (Farrukh 2025) | XMF-GNN (Ma 2025) |
|---|---|---|
| **GNN backbone** | GATConv (eq. 5-7 trong paper XG-NID) | SAGEConv (eq. 4-7 paper XMF, Sec. 3.3) |
| **Edge features** | Giữ 2 edge type `contain` (4 attrs) + `link` (1 attr) | **Loại bỏ** edge attributes; gấp vào node features; chỉ giữ 1 relation `connected_to` (Sec. 3.2) |
| **Temporal features** | 16 rolling-window features (XG-NID Table 1) — single-scale | **52 multi-scale** features {10s,30s,60s,300s} × 13 (Sec. 3.1, Alg. 1) |
| **Fusion strategy** | Concat + MLP (static) | **Attention-driven dynamic** fusion (Sec. 3.4, eq. 10-12) |
| **Explainability** | IG + LLM zero-shot (Sec. 3.1.5-3.1.6) | IG + ablation analysis (Sec. 4.6.4); không có LLM stage |

Vì thế file [Utility/Model.py](Utility/Model.py) **không tái sử dụng** lớp
`HeteroGNN` của XG-NID — toàn bộ kiến trúc được viết lại từ đầu để phản
ánh paper XMF-GNN.

---

## 4. Chi tiết từng component

### 4.1. Component 1 — Flow & Packet Feature Generator (Sec. 3.1)

**Paper:**
- NFStream với `idle_timeout=120`, max 20 packets / flow.
- 76 flow-level features (NFStream output sau khi drop meta).
- 14 packet-level protocol features.
- Mỗi packet → payload vector 1500 dims (mỗi byte hex → int 0-255, zero-pad).

**Code:** [Utility/Feature_extractor_flow_packet_combined.py](Utility/Feature_extractor_flow_packet_combined.py)
- Giữ nguyên `My_Custom(NFPlugin, limit=20)` của XG-NID — paper XMF-GNN trực
  tiếp citing hành vi này (Sec. 3.1: "1500-dimensional payload encoding ...
  follows the approach proposed by Farrukh et al. [6]").
- Output: CSV với cột `udps.payload_data`, `udps.packet_direction`,
  `udps.ip_size`, `udps.transport_size`, `udps.payload_size`, `udps.delta_time`,
  + 8 TCP flag bits (`udps.syn`, `udps.cwr`, `udps.ece`, `udps.urg`,
  `udps.ack`, `udps.psh`, `udps.rst`, `udps.fin`).

### 4.2. Component 2 — Multi-scale Temporal Feature Extractor (Sec. 3.1, Algorithm 1)

**Paper Algorithm 1:**
```
Input:  W = {w_1, ..., w_n}                — set of sliding window sizes
        F                                  — conventional flow features
Output: E = multi-scale enriched features

Step 1: For each destination D_j and each w ∈ W,
        initialize buffer W_j[w] = [0,...,0] (length = w).
Step 2: For each time step t_i, each destination D_j, each w ∈ W:
        - Update W_j[w] with packets received at t_i.
        - Compute temporal features T_j[w], such as:
            packet rate, average packet size, TCP SYN ratio,
            directional imbalance, ICMP proportion, RTT variance, etc.
Step 3: For each destination D_j, concatenate:
            E_j ← Combine(F, T_j[w_1], ..., T_j[w_n])
```

Paper Sec. 3.1 trích dẫn cụ thể *"computing behavioral features across
several sliding windows in parallel ... 10 s, 30 s, 60 s, 300 s ..."*.

Paper Sec. 3.2 ghi rõ flow node feature có **52 temporal features** =>
**13 features per window × 4 windows = 52**.

**Code:** [Utility/Additional_Features.py](Utility/Additional_Features.py)
- 13 features mỗi window:
  ```
  pkt_rate, byte_rate, avg_pkt_size, syn_ratio, ack_ratio,
  fin_rst_ratio, icmp_ratio, udp_ratio, src_dst_ratio,
  iat_mean, iat_std, unique_dst_ports, vuln_port_hits
  ```
- Chọn các metric phản ánh tất cả các ví dụ paper liệt kê (packet rate,
  avg packet size, SYN ratio, directional imbalance, ICMP proportion, RTT
  variance "etc.") + thêm 4 metric thường gặp trong literature NIDS
  (byte rate, ACK ratio, unique port spread, vuln-port hits).
- `_compute_per_destination_window` xử lý buffer per-destination IP với
  sliding window theo timestamp `bidirectional_first_seen_ms`.
- Output cột: `pkt_rate_10s`, `pkt_rate_30s`, ..., `vuln_port_hits_300s`
  → 52 cột mới được thêm vào CSV.

### 4.3. Component 3 — Heterogeneous Graph Construction (Sec. 3.2)

**Paper điểm chính:**
> "we eliminate the explicit representation of both edge types and instead
> retain only the connection relationships between the two types of nodes.
> The original edge features are embedded directly into the corresponding
> node attributes."

→ Bỏ 2 edge type `contain` và `link` của XG-NID. Chỉ còn duy nhất một
relation `connected_to`.

**Code:** [Utility/Functions.py:NIDSDataset._get_packet_node_features](Utility/Functions.py)
- Packet node feature 14 dims (paper Sec. 3.2):
  ```
  [direction, ip_size, transport_size, payload_size, delta_time,
   syn, cwr, ece, urg, ack, psh, rst, fin, payload_density]
  ```
  Trong đó:
  - 5 features đầu chính là **edge attributes của contain edge + link edge** đã
    được fold vào packet node:
    - `direction`, `ip_size`, `transport_size`, `payload_size` từ contain edge
      (4 attrs);
    - `delta_time` từ link edge (1 attr).
  - 8 TCP flag bits.
  - 1 feature derived `payload_density = payload_size / ip_size`.
  → tổng cộng **14 features** đúng paper.
- `_get_connected_edge_index`: edge index `(2, n_packets)` cho relation duy nhất.
- `T.ToUndirected()`: thêm reverse relation để message passing chạy 2 chiều
  (cần thiết với PyG; không vi phạm paper vì paper định nghĩa `E_{f,p}` ở
  eq. 3 mà không pin direction).

### 4.4. Component 4 — HGNN SAGE Encoder (Sec. 3.3)

**Paper eq. 4-7:**
```
h^(1)_v_i = σ(SAGEConv(h^(0)_v_i, A, E))                              (eq. 4)
σ(x) = LeakyReLU(x), α = 0.01                                          (eq. 5)
h^(l)_v_i = σ(W_α · h^(l)_v_i + (1/|N(v_i)|) Σ_{v_j ∈ N(v_i)} W_β · h^(l-1)_v_j)  (eq. 6)
BN(x) = ((x - μ)/sqrt(σ² + ε)) · γ + β                                  (eq. 7)
```

**Code:** [Utility/Model.py:HeteroSAGEEncoder](Utility/Model.py)
- 2 lớp `HeteroConv({et: SAGEConv((-1,-1), hidden) for et in edge_types})`,
  mỗi lớp đi kèm `BatchNorm1d` (mỗi node type) + `LeakyReLU(α=0.01)`.
- Per-node-type BN → tránh shift distribution giữa flow và packet nodes
  (paper eq. 7 nói rõ "Batch Normalization independently to each node type").
- `aggr='mean'` mặc định (paper Sec. 3.3 dùng "mean aggregator").

### 4.5. Component 5 — Per-modality Global Mean Pool (eq. 8-9)

**Paper eq. 8-9:**
```
h_f^(k) = (1/|V_f^(k)|) Σ_{v ∈ V_f^(k)} h^(2)_v
h_p^(k) = (1/|V_p^(k)|) Σ_{v ∈ V_p^(k)} h^(2)_v
```

**Code:** [Utility/Model.py:XMFGNN.encode](Utility/Model.py)
```python
h_f = pyg_nn.global_mean_pool(x_dict['flow'], batch.batch_dict['flow'])
h_p = pyg_nn.global_mean_pool(x_dict['packet'], batch.batch_dict['packet'])
```

### 4.6. Component 6 — Modality Attention Fusion (Sec. 3.4 / eq. 10-12 / Algorithm 2)

**Paper Algorithm 2:**
```
z_f ← W_f · h_f + b_f                                                  (eq. 10)
z_p ← W_p · h_p + b_p
for each sample i in batch:
    Z_i ← concat(z_f[i], z_p[i])                                       ∈ R^{2 × d'}
    α_i ← softmax(W_a · tanh(Z_i))                                     ∈ R^{2 × 1}  (eq. 11)
    α_f ← α_i[0], α_p ← α_i[1]
    h_ζ ← α_f · h_f[i] + α_p · h_p[i]                                  (eq. 12)
return h_ζ
```

**Lưu ý quan trọng (eq. 12):**
- Weighted sum dùng **`h_f` / `h_p` gốc**, không phải `z_f` / `z_p` (chỉ
  dùng cho scorer attention).
- Vì thế output dim = `hidden_size` (bằng dim đầu vào modality), không phải
  `attn_size`.

**Code:** [Utility/Model.py:ModalityFusion](Utility/Model.py)
```python
z_f = self.proj_f(h_f)              # (B, attn_size)
z_p = self.proj_p(h_p)
Z   = torch.stack([z_f, z_p], dim=1)        # (B, 2, attn_size)
scores = self.attn(torch.tanh(Z))           # (B, 2, 1)  -- shared scorer
alpha  = F.softmax(scores, dim=1)
h_zeta = alpha[:, 0] * h_f + alpha[:, 1] * h_p     # (B, hidden_size)
```
- `self.attn = nn.Linear(attn_size, 1, bias=False)` đúng "shared attention
  mechanism" (paper Sec. 3.4: *"a single parameter matrix W_a"*).
- `tanh(Z)` đúng *"non-linear mapping via the tanh function"*.
- `softmax(... dim=1)`: softmax giữa 2 modality (paper eq. 11).
- Lưu `_last_alpha` để vẽ Fig. 8 (flow attention vs epoch).

### 4.7. Component 7 — Classifier Head (eq. 13)

**Paper eq. 13:**
```
Out = LogSoftmax(W_2 · ReLU(W_1 · ReLU(W_0 · h_ζ)))
```

**Code:** [Utility/Model.py:XMFGNN.forward](Utility/Model.py)
```python
h = F.relu(self.cls_W0(h_zeta))
h = F.relu(self.cls_W1(h))
out = self.cls_W2(h)
return F.log_softmax(out, dim=-1)
```

### 4.8. Component 8 — Ablation Variants (Sec. 4.6.3 / Table 8)

| Variant | Paper mô tả | Code class |
|---|---|---|
| **B-GNN** | "constructs a graph solely based on flow features and performs intrusion detection without introducing any modality fusion mechanism" | `XMFGNN(fusion='baseline')` — flow-only, không có ModalityFusion |
| **FM-GNN** | "incorporates a simple modality fusion module that uses an MLP to perform weighted fusion of flow and packet features" | `MLPFusion` |
| **SA-GNN** | "structurally simpler attention mechanism, where modality fusion weights are computed through a learnable linear layer" | `SimpleAttnFusion` |
| **GA-GNN** | "introduces a gating mechanism to enhance controllability over attention distribution" | `GatedFusion` |
| **MA-GNN** | "extends FM-GNN by integrating a multi-head attention mechanism, which enables parallel learning of diverse fusion patterns via multiple attention subspaces" | `MultiHeadAttnFusion` (default 4 heads) |
| **XMF-GNN** | Full paper architecture | `ModalityFusion` |

### 4.9. Component 9 — Training Pipeline (Sec. 4.4 / Table 3)

**Paper Table 3:**
```
Optimizer       Adam
Loss            NLL
Batch size      64
Learning rate   [10^-2, 10^-5]
Epoch size      100
Hidden size     64
Attn size       32
Pooling method  Mean
```

**Paper Sec. 4.4:** *"learning rate is dynamically adjusted via the
ReduceLROnPlateau scheduler ... a decay is triggered when the validation
metric improves by less than 0.01 for five consecutive epochs"*.

**Code:** [Utility/Training.py:make_optimizer_and_scheduler](Utility/Training.py)
```python
optim = torch.optim.Adam(model.parameters(), lr=1e-2)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optim, mode='max', factor=0.5,
    patience=5, threshold=0.01, min_lr=1e-5)
```

### 4.10. Component 10 — Integrated Gradient Explainer (Sec. 4.6.4 / eq. 19)

**Paper eq. 19:**
```
IG_i(x) = (x_i - x'_i) · ∫_{α=0}^{1} (∂F(x' + α(x - x'))/∂x_i) dα
```

**Code:** [Utility/IG_Explainer.py](Utility/IG_Explainer.py)
- Riemann sum approximation với `n_steps=50` (default).
- Apply riêng cho `flow` features và `packet` features.
- Baseline mặc định = zero (paper *"If a baseline input is not provided,
  zero is used as the default value"*, theo Sundararajan 2017).
- Helpers `top_flow_features`, `top_packet_features`, `top_payload_bytes`
  để reproduce paper Fig. 6 (Flow Node Feature Importance) và Fig. 7
  (Packet Node Feature Importance).

---

## 5. Danh sách file đã tạo

| File | Vai trò |
|---|---|
| [Utility/__init__.py](Utility/__init__.py) | Re-export các symbol công khai |
| [Utility/Model.py](Utility/Model.py) | `HeteroSAGEEncoder`, `ModalityFusion`, `MLPFusion`, `SimpleAttnFusion`, `GatedFusion`, `MultiHeadAttnFusion`, `XMFGNN` |
| [Utility/Functions.py](Utility/Functions.py) | `NIDSDataset`, `split_csv`, `Combining_classes`, `standardize_flow_features`, attacker MAC list, label dict |
| [Utility/Additional_Features.py](Utility/Additional_Features.py) | `additional_features` (multi-scale Algorithm 1), `window_feature_names`, `PER_WINDOW_FEATURE_NAMES` |
| [Utility/Feature_extractor_flow_packet_combined.py](Utility/Feature_extractor_flow_packet_combined.py) | NFStream extractor (CLI + `My_Custom` plugin) |
| [Utility/Training.py](Utility/Training.py) | `train`, `test`, `test_cm`, `make_optimizer_and_scheduler`, `collect_flow_attention` |
| [Utility/IG_Explainer.py](Utility/IG_Explainer.py) | `IntegratedGradientExplainer`, helpers `top_flow_features`, `top_packet_features`, `top_payload_bytes` |
| [XMF_GNN.ipynb](XMF_GNN.ipynb) | Pipeline PCAP → CSV → multi-scale → graph dataset |
| [XMF_GNN_Model.ipynb](XMF_GNN_Model.ipynb) | Train, evaluate, ablation, IG explainer |
| [Data_preprocessing_CIC-IoT2023.ipynb](Data_preprocessing_CIC-IoT2023.ipynb) | MAC filter, balance, train/test split |
| [requirements.txt](requirements.txt) | torch 2.3.1+cu118, PyG 2.5.3, NFStream 6.5.3, ... |
| [README.md](README.md) | Quickstart, repo layout, expected results |

---

## 6. Cách chạy lại

### 6.1. Yêu cầu môi trường
- **OS:** Linux/macOS hoặc Windows + WSL2 (NFStream).
- **Python:** 3.8 – 3.10.
- **GPU:** Paper Sec. 4.4: 2× NVIDIA RTX 3090 (24 GB VRAM mỗi cái). Single
  GPU 12-16 GB cũng chạy được nhưng cần giảm batch size.
- **Disk:** ~100 GB free (CIC-IoT2023 raw pcap ~70 GB + processed graphs).

### 6.2. Cài đặt

```bash
conda create -n xmfgnn python=3.10 -y
conda activate xmfgnn
pip install -r requirements.txt
```

### 6.3. Pipeline đầy đủ

1. **PCAP → CSV** — chạy `XMF_GNN.ipynb` cell "Step 1".
2. **Add multi-scale features** — `XMF_GNN.ipynb` cell "Step 2".
3. **Filter + balance** — `Data_preprocessing_CIC-IoT2023.ipynb`.
4. **Build graph dataset** — `XMF_GNN.ipynb` cell "Step 4".
5. **Train + evaluate** — `XMF_GNN_Model.ipynb`.
6. **Ablation** — section 2 của `XMF_GNN_Model.ipynb`.
7. **IG explainer** — section 4 của `XMF_GNN_Model.ipynb`.

### 6.4. Kết quả mong đợi

Paper Tables 4 / 5 (multi-class / binary classification):

| Dataset | Task | Accuracy | F1 |
|---|---|---|---|
| CIC-IDS2017 | multi | 0.9743 | **0.9774** |
| CIC-IDS2017 | binary | 0.9872 | **0.9878** |
| CIC-IoT2023 | multi | 0.9843 | **0.9852** |
| CIC-IoT2023 | binary | 0.9983 | **0.9979** |

Per-class recall/F1 (paper Table 6, CIC-IoT2023, multi-class):

| Class | Recall | F1 |
|---|---|---|
| Benign | 0.9984 | 0.9876 |
| WebBased | 0.9973 | 0.9634 |
| Spoofing | 0.9876 | 0.9846 |
| Recon | 0.9341 | 0.9683 |
| Mirai | 1.0000 | 1.0000 |
| DoS | 1.0000 | 1.0000 |
| DDoS | 1.0000 | 1.0000 |
| BruteForce | 1.0000 | 1.0000 |

---

## 7. Phụ lục: bảng tham chiếu paper ↔ code

### 7.1. Paper Sec. 3.1 / Algorithm 1 — Multi-scale Feature Extractor

| Paper | Code |
|---|---|
| `W = {w_1, ..., w_n}` (sliding window sizes) | `DEFAULT_WINDOW_SIZES_SEC = (10, 30, 60, 300)` |
| Step 1: init buffers per destination | `_compute_per_destination_window` (groups by `dst_ip`) |
| Step 2: update buffer + compute per-window features | sliding deque trong `_compute_per_destination_window` |
| Step 3: concat F + T_j[w_1] + ... + T_j[w_n] | Loop `for w in window_sizes_sec: data[f"{name}_{w}s"] = ...` |
| Output: 52-dim block | 13 × 4 = 52 cột mới |

### 7.2. Paper Sec. 3.2 — Graph Construction

| Paper | Code |
|---|---|
| Flow node `V_f`: 76 base + 52 multi-scale = 128 | `_get_flow_node_features` |
| Packet node `V_p`: 1500 payload + 14 protocol = 1514 | `_get_packet_node_features` |
| Single relation `connected_to` (eq. 3: `V = ({V_f ∪ V_p}, E_{f,p})`) | `data['flow', 'connected_to', 'packet'].edge_index` |
| Edge attributes folded into nodes | 5 numeric (direction/ip_size/...) + 8 flags + 1 derived = 14 packet features |

### 7.3. Paper Sec. 3.3 / eq. 4-7 — HGNN Encoder

| Paper | Code |
|---|---|
| eq. 4: `h^(1) = σ(SAGEConv(h^(0), A, E))` | `self.conv1(x_dict, edge_index_dict)` |
| eq. 5: σ = LeakyReLU(α=0.01) | `nn.LeakyReLU(negative_slope=0.01)` |
| eq. 6: SAGEConv update rule với `W_α` (self) + `W_β` (neighbor) | PyG `SAGEConv` mặc định đúng formulation này |
| eq. 7: BN per node type | `self.bn1[k]`, `self.bn2[k]` per node type |
| 2-layer architecture | `self.conv1` + `self.conv2` |
| Mean aggregator | `aggr='mean'` |

### 7.4. Paper Sec. 3.4 / Algorithm 2 / eq. 10-12 — Modality Fusion

| Paper | Code (`ModalityFusion`) |
|---|---|
| eq. 10: `z = W h + b` | `self.proj_f`, `self.proj_p` (Linear with bias) |
| Stack to Z ∈ R^{B × 2 × d'} | `torch.stack([z_f, z_p], dim=1)` |
| Shared scorer `W_a` | `self.attn = nn.Linear(attn_size, 1, bias=False)` |
| eq. 11: `α = softmax(W_a · tanh(Z))` | `F.softmax(self.attn(torch.tanh(Z)), dim=1)` |
| eq. 12: `h_ζ = α_f · h_f + α_p · h_p` (dùng h gốc, không z) | `alpha[:, 0] * h_f + alpha[:, 1] * h_p` |

### 7.5. Paper eq. 13 — Classifier Head

| Paper | Code |
|---|---|
| `Out = LogSoftmax(W_2 · ReLU(W_1 · ReLU(W_0 · h_ζ)))` | `F.log_softmax(self.cls_W2(F.relu(self.cls_W1(F.relu(self.cls_W0(h_zeta)))))` |

### 7.6. Paper Sec. 4.6.3 / Table 8 — Ablation

| Variant | Paper accuracy | Code |
|---|---|---|
| B-GNN | 0.9578 | `XMFGNN(fusion='baseline')` |
| FM-GNN | 0.9759 | `XMFGNN(fusion='mlp')` |
| SA-GNN | 0.9792 | `XMFGNN(fusion='simple_attn')` |
| GA-GNN | 0.9810 | `XMFGNN(fusion='gated')` |
| MA-GNN | 0.9840 | `XMFGNN(fusion='multi_head')` |
| **XMF-GNN** | **0.9843** | `XMFGNN(fusion='attn')` |

### 7.7. Paper Sec. 4.6.4 / eq. 19 — Integrated Gradient

| Paper | Code |
|---|---|
| `IG_i(x) = (x_i - x'_i) · ∫_{α=0}^{1} ∂F/∂x_i dα` | `IntegratedGradientExplainer.explain` (Riemann sum n_steps điểm) |
| Baseline = 0 default | `flow_baseline = torch.zeros_like(flow_x)` |
| Apply riêng cho flow + packet features | 2 dòng `flow_grad_sum`, `packet_grad_sum` |
| Fig. 6 (top flow features) | `top_flow_features(attr, FLOW_FEATURE_NAMES, top_n=10)` |
| Fig. 7 (top packet features) | `top_packet_features(attr, PACKET_PROTO_NAMES, top_n=10)` |

### 7.8. Paper Sec. 4.2 / Tables 1-3 — Data Preprocessing

| Paper | Code |
|---|---|
| Bidirectional MAC filter (9 attacker MACs) | `CIC_IOT2023_ATTACKER_MACS` + `split_csv` |
| Train: 20,000 samples / class × 8 classes | `Combining_classes(Number_in_individaul_class=20000)` |
| Test: 4,000 cho class lớn, ~20% cho class hiếm | `split_csv` với threshold 35,000 + `frac=0.2` fallback |
| Over/under-sampling cho minority class | `duplicate_rows` + `random_pick_rows` |
| Standardization eq. 14: `x' = (x - μ)/σ` | `standardize_flow_features` |

### 7.9. Paper Sec. 4.4 / Table 3 — Hyperparameters

| Paper | Code |
|---|---|
| optimizer = Adam | `torch.optim.Adam` |
| loss = NLL | `F.nll_loss` |
| batch_size = 64 | `args['batch_size'] = 64` |
| lr ∈ [1e-2, 1e-5] | `lr=1e-2`, `min_lr=1e-5` |
| epoch = 100 | `args['epochs'] = 100` |
| hidden = 64 | `args['hidden_size'] = 64` |
| attn = 32 | `args['attn_size'] = 32` |
| pooling = mean | `pyg_nn.global_mean_pool` |
| ReduceLROnPlateau patience=5, threshold=0.01 | `make_optimizer_and_scheduler(patience=5, threshold=0.01)` |

---

## 8. Các điểm paper không cố định và quyết định thiết kế

Paper có một số chỗ không nêu chính xác chi tiết implementation; những lựa
chọn chúng tôi áp dụng (kèm justification từ paper):

### 8.1. Chính xác 13 features mỗi window

Paper Algorithm 1 chỉ liệt kê *"such as: Packet rate (pkt/s), Average
packet size, TCP SYN ratio, Directional imbalance (src/dst ratio), ICMP
proportion, RTT variance, etc."* — tức 6 ví dụ + "etc.". Paper Sec. 3.2
ghi rõ flow node feature có 52 temporal features. Vì 4 windows × `k` =
52 → `k = 13`.

**Thiết kế chọn:** 13 features = 6 example trong Algorithm 1 + 7 metric
phổ biến NIDS (byte_rate, ack_ratio, fin_rst_ratio, udp_ratio, iat_mean,
iat_std, unique_dst_ports, vuln_port_hits — bỏ RTT vì NFStream không
cung cấp RTT mà chỉ có inter-arrival).

**Mức độ ảnh hưởng:** Trung bình. Nếu paper có ý dùng feature khác, kết
quả có thể lệch ±0.5% F1. User có thể edit `PER_WINDOW_FEATURE_NAMES`
trong [Utility/Additional_Features.py](Utility/Additional_Features.py) để
thử bộ feature khác.

### 8.2. Direction của edge `connected_to`

Paper eq. 3: `V = ({V_f ∪ V_p}, E_{f,p})` — không pin direction.
**Thiết kế:** flow → packet, sau đó `T.ToUndirected()` thêm reverse. Đây
là idiom chuẩn của PyG, cần thiết để message passing 2 chiều.

### 8.3. Số attention heads cho MA-GNN

Paper Sec. 4.6.3: *"multi-head attention mechanism, which enables parallel
learning of diverse fusion patterns via multiple attention subspaces"* —
không cố định số head.

**Thiết kế:** mặc định 4 heads (xem `MultiHeadAttnFusion(n_heads=4)`).
Có thể đổi qua `XMFGNN(..., fusion='multi_head', n_heads=8)`.

### 8.4. BatchNorm eps

Paper eq. 7 dùng ε generic. **Thiết kế:** dùng PyTorch default `eps=1e-5`.

### 8.5. Edge feature → packet node attribute (folding strategy)

Paper Sec. 3.2 nói rõ "embed edge features into corresponding node
attributes" nhưng không nêu cách cụ thể.
**Thiết kế:**
- Contain edge attributes (4 attrs: direction, ip_size, transport_size,
  payload_size) đều là **per-packet** properties → fold vào *packet node*.
- Link edge attribute (delta_time, 1 attr) cũng per-packet (giữa packet i
  và i-1) → fold vào *packet node* với delta của packet đó với packet
  trước.
- Không fold vào flow node vì các attrs này không có ý nghĩa flow-level.

Để giữ đúng số 14 packet features (paper Sec. 3.2), thêm 1 derived
`payload_density = payload_size / ip_size` (kết hợp 2 trong các attrs
edge gốc, không tạo ra information mới).

### 8.6. Window timestamp anchor

Paper Algorithm 1 không nêu rõ window neo từ `bidirectional_first_seen_ms`,
`bidirectional_last_seen_ms` hay `mid`.
**Thiết kế:** dùng `bidirectional_first_seen_ms` (tương đương với XG-NID
extractor và phù hợp với "for each time step t_i" của Algorithm 1).

### 8.7. Random seed

Paper không nêu seed. **Thiết kế:** seed=42 cho tất cả `random_state`
trong preprocessing (sample, train_test_split). Training không cố định
seed (paper Sec. 4.6.1 báo *"average results"* qua nhiều lần lặp).

---

## Liên hệ & ghi chú

- **Paper gốc:** Neurocomputing 655 (2025) 131285,
  [DOI: 10.1016/j.neucom.2025.131285](https://doi.org/10.1016/j.neucom.2025.131285).
- **GNN4ID (XG-NID baseline):** https://github.com/Yasir-ali-farrukh/GNN4ID
- **Author email (paper):** mzx@zua.edu.cn (Z. Ma — corresponding).

**Trạng thái implementation:**
- Code đã viết theo paper, **chưa chạy benchmark thực tế** trên CIC-IoT2023
  (chưa có GPU + dataset trong môi trường phát triển hiện tại).
- Sau khi chạy thực tế, nếu F1 lệch nhiều với paper, debug theo thứ tự:
  1. Bộ 13 features per-window (Sec. 8.1).
  2. `window_size` định nghĩa (sliding window theo packet count vs
     theo seconds — paper không nói rõ hoàn toàn; chúng tôi chọn seconds).
  3. Hidden size / batch size / lr.
  4. Random seed.
