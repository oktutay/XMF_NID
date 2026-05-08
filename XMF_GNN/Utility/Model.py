"""
XMF-GNN Model — paper Section 3.3 (HGNN) + Section 3.4 (ModalityFusion).

This file implements the architecture described in:

    Ma, Liu, Chen, Liu, Li.  "XMF-GNN: A cross-modality dynamic fusion
    heterogeneous graph neural network for network intrusion detection."
    Neurocomputing 655 (2025) 131285.

Model overview (paper Fig. 1):

    HeteroData (flow + packet, single edge type "connected_to")
        |
        v
    HGNN (2-layer SAGEConv encoder, paper eq. 4-7)
        - SAGEConv (per node type) -> BN (per node type) -> LeakyReLU
        - SAGEConv (per node type) -> BN (per node type) -> LeakyReLU
        |
        v
    Per-modality global mean pool      (paper eq. 8-9)
        - h_f = mean_{v in V_f^(k)} h_v^(2)
        - h_p = mean_{v in V_p^(k)} h_v^(2)
        |
        v
    ModalityFusion (paper Section 3.4 + Algorithm 2)
        - Linear projection to a shared d' hidden space
            z_f = W_f h_f + b_f,  z_p = W_p h_p + b_p
        - Stack -> Z in R^{B x 2 x d'}
        - alpha = softmax(W_a . tanh(Z))   (shared attention scorer)
        - h_zeta = alpha_f * h_f + alpha_p * h_p          (NOT z_f / z_p)
        |
        v
    Classifier head (paper eq. 13)
        Out = LogSoftmax(W2 . ReLU(W1 . ReLU(W0 . h_zeta)))

Edge attributes are intentionally NOT consumed: paper Sec. 3.2 explicitly
removes the explicit "contain" / "link" edge types and folds those features
into the corresponding node attributes, leaving a single "connected_to"
relation. ToUndirected() in the dataset adds the reverse relation so message
passing flows both ways.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import HeteroConv, SAGEConv


# ---------------------------------------------------------------------------
# 1. Two-layer heterogeneous SAGE encoder (paper Sec. 3.3, eq. 4-7).
# ---------------------------------------------------------------------------


class HeteroSAGEEncoder(nn.Module):
    """Two-layer GraphSAGE encoder over a heterogeneous graph.

    For every directed edge type in ``hetero_metadata[1]`` (e.g.
    ``('flow','connected_to','packet')`` and its reverse) we register a
    SAGEConv. PyG's ``HeteroConv`` then dispatches messages per relation and
    aggregates with the configured ``aggr`` (default 'mean').

    Per-node-type BatchNorm and LeakyReLU follow each layer (paper eq. 6-7
    explicitly mentions BN and the LeakyReLU activation with alpha=0.01).
    """

    def __init__(self, hetero_metadata, hidden_size: int,
                 aggr: str = "mean", bn_eps: float = 1.0,
                 leaky_slope: float = 0.01):
        super().__init__()
        self.hidden_size = hidden_size

        node_types, edge_types = hetero_metadata

        self.conv1 = HeteroConv(
            {et: SAGEConv((-1, -1), hidden_size, aggr=aggr) for et in edge_types},
            aggr='sum',
        )
        self.conv2 = HeteroConv(
            {et: SAGEConv((-1, -1), hidden_size, aggr=aggr) for et in edge_types},
            aggr='sum',
        )

        self.bn1 = nn.ModuleDict(
            {nt: nn.BatchNorm1d(hidden_size, eps=bn_eps) for nt in node_types}
        )
        self.bn2 = nn.ModuleDict(
            {nt: nn.BatchNorm1d(hidden_size, eps=bn_eps) for nt in node_types}
        )
        self.act = nn.LeakyReLU(negative_slope=leaky_slope)

    def forward(self, x_dict, edge_index_dict):
        # Layer 1: SAGE -> BN -> LeakyReLU
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: self.bn1[k](v) for k, v in x_dict.items()}
        x_dict = {k: self.act(v) for k, v in x_dict.items()}

        # Layer 2: SAGE -> BN -> LeakyReLU
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: self.bn2[k](v) for k, v in x_dict.items()}
        x_dict = {k: self.act(v) for k, v in x_dict.items()}
        return x_dict


# ---------------------------------------------------------------------------
# 2. Modality fusion modules (paper Sec. 3.4 + ablation variants Sec. 4.6.3).
# ---------------------------------------------------------------------------


class ModalityFusion(nn.Module):
    """Attention-driven dynamic modality fusion (paper Algorithm 2 / eq. 10-12).

    Steps:
        1. Project per-modality graph embeddings to a shared d' space:
              z_f = W_f h_f + b_f       (paper eq. 10)
              z_p = W_p h_p + b_p
           Stacked across the modality dimension to Z in R^{B x 2 x d'}.
        2. Compute attention weights with a shared scorer:
              alpha = softmax(W_a . tanh(Z))    in R^{B x 2 x 1}    (paper eq. 11)
        3. Weighted combination *of the original modality embeddings*:
              h_zeta = alpha_f * h_f + alpha_p * h_p                 (paper eq. 12)

    Note: paper eq. 12 uses ``h_f`` / ``h_p``, not the projected ``z_f`` /
    ``z_p`` -- the projection only feeds the attention scorer. We keep that
    convention so fused dim equals ``hidden_size`` (the input modality dim).
    """

    def __init__(self, hidden_size: int, attn_size: int):
        super().__init__()
        self.proj_f = nn.Linear(hidden_size, attn_size, bias=True)
        self.proj_p = nn.Linear(hidden_size, attn_size, bias=True)
        # Shared attention scorer W_a: R^{d' -> 1}.
        self.attn = nn.Linear(attn_size, 1, bias=False)

        self._last_alpha: Optional[torch.Tensor] = None  # for diagnostics

    def forward(self, h_f: torch.Tensor, h_p: torch.Tensor):
        z_f = self.proj_f(h_f)              # (B, attn_size)
        z_p = self.proj_p(h_p)              # (B, attn_size)
        Z = torch.stack([z_f, z_p], dim=1)  # (B, 2, attn_size)

        scores = self.attn(torch.tanh(Z))   # (B, 2, 1)
        alpha = F.softmax(scores, dim=1)    # softmax across modalities
        self._last_alpha = alpha.detach()

        alpha_f = alpha[:, 0, :]            # (B, 1)
        alpha_p = alpha[:, 1, :]            # (B, 1)
        h_zeta = alpha_f * h_f + alpha_p * h_p
        return h_zeta, alpha


class SimpleAttnFusion(nn.Module):
    """SA-GNN ablation (paper Table 8): a single learnable linear layer
    produces fusion weights from concat([h_f, h_p]).

    weights = softmax(W [h_f ; h_p])  in R^{B x 2}
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.scorer = nn.Linear(2 * hidden_size, 2, bias=True)
        self._last_alpha: Optional[torch.Tensor] = None

    def forward(self, h_f, h_p):
        combined = torch.cat([h_f, h_p], dim=-1)
        weights = F.softmax(self.scorer(combined), dim=-1)  # (B, 2)
        self._last_alpha = weights.detach().unsqueeze(-1)
        h_zeta = weights[:, 0:1] * h_f + weights[:, 1:2] * h_p
        return h_zeta, weights


class GatedFusion(nn.Module):
    """GA-GNN ablation (paper Table 8): gating mechanism with sigmoid gate.

    gate = sigmoid(W_g [h_f ; h_p])
    h_zeta = gate * h_f + (1 - gate) * h_p

    The gate is element-wise so each fused dimension can mix the modalities
    independently, mirroring the typical GLU / Highway gating literature.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.gate = nn.Linear(2 * hidden_size, hidden_size, bias=True)
        self._last_alpha: Optional[torch.Tensor] = None

    def forward(self, h_f, h_p):
        combined = torch.cat([h_f, h_p], dim=-1)
        g = torch.sigmoid(self.gate(combined))     # (B, hidden_size)
        h_zeta = g * h_f + (1.0 - g) * h_p
        # Diagnostic: average gate value treated as alpha_f.
        self._last_alpha = torch.stack(
            [g.mean(dim=-1, keepdim=True), (1 - g).mean(dim=-1, keepdim=True)],
            dim=1,
        ).detach()
        return h_zeta, g


class MLPFusion(nn.Module):
    """FM-GNN ablation (paper Table 8): plain MLP fusion with no attention.

    h_zeta = MLP(concat([h_f, h_p]))
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self._last_alpha: Optional[torch.Tensor] = None

    def forward(self, h_f, h_p):
        h_zeta = self.mlp(torch.cat([h_f, h_p], dim=-1))
        return h_zeta, None


class MultiHeadAttnFusion(nn.Module):
    """MA-GNN ablation (paper Table 8 / Fig. 8a): multi-head self-attention
    fusion with `n_heads` heads. Each head learns an independent set of
    fusion weights; final embedding is the mean across heads.
    """

    def __init__(self, hidden_size: int, attn_size: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.proj_f = nn.Linear(hidden_size, attn_size * n_heads, bias=True)
        self.proj_p = nn.Linear(hidden_size, attn_size * n_heads, bias=True)
        self.attn = nn.Linear(attn_size, 1, bias=False)
        self.attn_size = attn_size
        self._last_alpha: Optional[torch.Tensor] = None

    def forward(self, h_f, h_p):
        B = h_f.shape[0]
        z_f = self.proj_f(h_f).view(B, self.n_heads, self.attn_size)  # (B,H,d)
        z_p = self.proj_p(h_p).view(B, self.n_heads, self.attn_size)
        Z = torch.stack([z_f, z_p], dim=2)                            # (B,H,2,d)

        scores = self.attn(torch.tanh(Z))                             # (B,H,2,1)
        alpha = F.softmax(scores, dim=2)                              # over modalities
        self._last_alpha = alpha.detach()

        alpha_f = alpha[:, :, 0, :]   # (B, H, 1)
        alpha_p = alpha[:, :, 1, :]   # (B, H, 1)
        # Per-head weighted combination of the *original* embeddings, then
        # average across heads to keep fused dim = hidden_size.
        h_f_e = h_f.unsqueeze(1).expand(-1, self.n_heads, -1)
        h_p_e = h_p.unsqueeze(1).expand(-1, self.n_heads, -1)
        head_out = alpha_f * h_f_e + alpha_p * h_p_e   # (B, H, hidden)
        h_zeta = head_out.mean(dim=1)                  # (B, hidden)
        return h_zeta, alpha


# ---------------------------------------------------------------------------
# 3. Top-level XMF-GNN model.
# ---------------------------------------------------------------------------


class XMFGNN(nn.Module):
    """End-to-end XMF-GNN model.

    Args:
        hetero_metadata: ``HeteroData.metadata()`` -- (node_types, edge_types).
        hidden_size: hidden dim of the SAGE encoder and pre-fusion modality
            embeddings (paper Table 3 default = 64).
        attn_size: hidden dim of the modality attention scorer
            (paper Table 3 default = 32).
        num_classes: 8 for CIC-IoT2023 multi-class, 2 for binary, 15 for
            CIC-IDS2017 multi-class.
        fusion: which fusion variant to use. One of:
            'attn'        -- the paper's XMF-GNN (default).
            'mlp'         -- FM-GNN ablation.
            'simple_attn' -- SA-GNN ablation.
            'gated'       -- GA-GNN ablation.
            'multi_head'  -- MA-GNN ablation.
            'baseline'    -- B-GNN (flow modality only, no fusion module).
        aggr: SAGEConv neighborhood aggregator (default 'mean').
        bn_eps: BatchNorm eps (paper Table 3 hidden_size=64; eps not pinned,
            default 1e-5 in PyTorch).
        n_heads: heads for the multi-head fusion variant.
    """

    def __init__(self,
                 hetero_metadata,
                 hidden_size: int = 64,
                 attn_size: int = 32,
                 num_classes: int = 8,
                 fusion: str = "attn",
                 aggr: str = "mean",
                 bn_eps: float = 1e-5,
                 n_heads: int = 4):
        super().__init__()

        node_types, _ = hetero_metadata
        if 'flow' not in node_types or 'packet' not in node_types:
            raise ValueError(
                "XMF-GNN expects node types 'flow' and 'packet' in the "
                f"heterogeneous graph, got: {node_types}"
            )

        self.fusion_name = fusion
        self.encoder = HeteroSAGEEncoder(
            hetero_metadata, hidden_size, aggr=aggr, bn_eps=bn_eps,
        )

        if fusion == "attn":
            self.fusion = ModalityFusion(hidden_size, attn_size)
            fused_dim = hidden_size
        elif fusion == "mlp":
            self.fusion = MLPFusion(hidden_size)
            fused_dim = hidden_size
        elif fusion == "simple_attn":
            self.fusion = SimpleAttnFusion(hidden_size)
            fused_dim = hidden_size
        elif fusion == "gated":
            self.fusion = GatedFusion(hidden_size)
            fused_dim = hidden_size
        elif fusion == "multi_head":
            self.fusion = MultiHeadAttnFusion(hidden_size, attn_size, n_heads=n_heads)
            fused_dim = hidden_size
        elif fusion == "baseline":
            # B-GNN: flow-only, no fusion module.
            self.fusion = None
            fused_dim = hidden_size
        else:
            raise ValueError(f"Unknown fusion mode: {fusion!r}")

        # Classifier head (paper eq. 13):
        #   Out = LogSoftmax(W2 . ReLU(W1 . ReLU(W0 . h_zeta)))
        self.cls_W0 = nn.Linear(fused_dim, hidden_size)
        self.cls_W1 = nn.Linear(hidden_size, max(num_classes * 2, 16))
        self.cls_W2 = nn.Linear(max(num_classes * 2, 16), num_classes)

    # ------------------------------------------------------------------
    def encode(self, x_dict, edge_index_dict, batch):
        """Run the HGNN + per-modality global mean pool. Returns (h_f, h_p)."""
        x_dict = self.encoder(x_dict, edge_index_dict)
        # Per-modality graph-level pooling (paper eq. 8-9).
        # ``batch`` here is a Batch object holding ``batch_dict`` keyed by
        # node type, populated by PyG's DataLoader.
        h_f = pyg_nn.global_mean_pool(x_dict['flow'], batch.batch_dict['flow'])
        h_p = pyg_nn.global_mean_pool(x_dict['packet'], batch.batch_dict['packet'])
        return h_f, h_p

    def forward(self, x_dict, edge_index_dict, batch):
        h_f, h_p = self.encode(x_dict, edge_index_dict, batch)

        if self.fusion is None:
            # B-GNN baseline: flow-only.
            h_zeta = h_f
            alpha = None
        else:
            h_zeta, alpha = self.fusion(h_f, h_p)

        h = F.relu(self.cls_W0(h_zeta))
        h = F.relu(self.cls_W1(h))
        out = self.cls_W2(h)
        return F.log_softmax(out, dim=-1)

    def loss(self, preds, label):
        return F.nll_loss(preds, label)

    # Diagnostic helper for the attention-trace plot in Sec. 4.6.4 / Fig. 8.
    def get_last_alpha(self) -> Optional[torch.Tensor]:
        if self.fusion is None:
            return None
        return getattr(self.fusion, "_last_alpha", None)
