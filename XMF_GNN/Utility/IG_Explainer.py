"""
Integrated Gradient explainer for XMF-GNN, paper Section 4.6.4 / eq. (19).

Paper eq. (19):

    IG_i(x) = (x_i - x'_i) * integral_{alpha=0}^{1}
                  (dF / dx_i)|_{x' + alpha*(x - x')}  d_alpha

We approximate the integral with a Riemann sum over ``n_steps`` interpolation
points and apply the rule independently to flow-node features (paper Fig. 6)
and packet-node features (paper Fig. 7).

The XMF-GNN forward signature is ``(x_dict, edge_index_dict, batch)`` -- no
edge_attr_dict (paper Sec. 3.2 removes edge attributes). This explainer
respects that by passing only the two node-feature tensors as
gradient-tracked inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
from torch_geometric.data import Batch, HeteroData


@dataclass
class IGAttribution:
    """Single-sample IG result.

    Arrays returned as numpy so they can be sorted/printed/fed into prompt
    builders without further conversion.
    """
    predicted_class: int
    predicted_logprob: float
    flow_attr: np.ndarray         # (num_flow_features,)
    packet_attr: np.ndarray       # (num_packets, packet_feat_dim)
    flow_values: np.ndarray
    packet_values: np.ndarray


class IntegratedGradientExplainer:
    """Compute Integrated Gradient attributions for a single graph at a time.

    The XMF-GNN model expects a PyG ``Batch`` (single graphs are still
    bundled into a Batch of size 1 via ``Batch.from_data_list``). We mimic
    that here so the model's forward signature is untouched.
    """

    def __init__(self, model: torch.nn.Module, device: str = "cuda",
                 n_steps: int = 50):
        self.model = model
        self.device = device
        self.n_steps = n_steps
        self.model.eval()

    # ------------------------------------------------------------------
    def explain(self, data: HeteroData,
                target_class: Optional[int] = None,
                flow_baseline: Optional[torch.Tensor] = None,
                packet_baseline: Optional[torch.Tensor] = None) -> IGAttribution:
        """Compute IG attributions for ``data``.

        Args:
            data: A single PyG HeteroData (same shape as ``NIDSDataset[i]``).
            target_class: If None, computed wrt the predicted class.
            flow_baseline / packet_baseline: Reference inputs ``x'``. Default
                = zeros (paper Sec. 3.1.5 of XG-NID, also used by XMF-GNN
                Sec. 4.6.4: "If a baseline input is not provided, zero is
                used as the default value").
        """
        batch = Batch.from_data_list([data]).to(self.device)
        flow_x = batch["flow"].x.detach().clone()
        packet_x = batch["packet"].x.detach().clone()

        if flow_baseline is None:
            flow_baseline = torch.zeros_like(flow_x)
        else:
            flow_baseline = flow_baseline.to(self.device)
        if packet_baseline is None:
            packet_baseline = torch.zeros_like(packet_x)
        else:
            packet_baseline = packet_baseline.to(self.device)

        # Forward once at the actual input to get the predicted class.
        with torch.no_grad():
            logits = self._forward(batch, flow_x, packet_x)
            if target_class is None:
                target_class = int(logits.argmax(dim=1).item())
            target_logprob = float(logits[0, target_class].item())

        # Riemann-sum approximation of the integral in eq. (19) over
        # n_steps linearly spaced alphas.
        flow_grad_sum = torch.zeros_like(flow_x)
        packet_grad_sum = torch.zeros_like(packet_x)

        for k in range(1, self.n_steps + 1):
            alpha = k / self.n_steps
            f_interp = (flow_baseline + alpha * (flow_x - flow_baseline)).requires_grad_(True)
            p_interp = (packet_baseline + alpha * (packet_x - packet_baseline)).requires_grad_(True)

            logits = self._forward(batch, f_interp, p_interp)
            score = logits[0, target_class]

            grads = torch.autograd.grad(
                score, [f_interp, p_interp],
                retain_graph=False, create_graph=False, allow_unused=True,
            )
            g_f = grads[0] if grads[0] is not None else torch.zeros_like(flow_x)
            g_p = grads[1] if grads[1] is not None else torch.zeros_like(packet_x)
            flow_grad_sum = flow_grad_sum + g_f.detach()
            packet_grad_sum = packet_grad_sum + g_p.detach()

        avg_flow_grad = flow_grad_sum / self.n_steps
        avg_packet_grad = packet_grad_sum / self.n_steps

        flow_ig = (flow_x - flow_baseline) * avg_flow_grad
        packet_ig = (packet_x - packet_baseline) * avg_packet_grad

        return IGAttribution(
            predicted_class=target_class,
            predicted_logprob=target_logprob,
            flow_attr=flow_ig.detach().cpu().numpy().squeeze(0),
            packet_attr=packet_ig.detach().cpu().numpy(),
            flow_values=flow_x.detach().cpu().numpy().squeeze(0),
            packet_values=packet_x.detach().cpu().numpy(),
        )

    # ------------------------------------------------------------------
    def _forward(self, original_batch: Batch,
                 flow_x: torch.Tensor, packet_x: torch.Tensor) -> torch.Tensor:
        """Rebuild a Batch with patched node-feature tensors and call the
        model. Edge indices / batch_dict are reused from ``original_batch``.
        """
        x_dict = {"flow": flow_x, "packet": packet_x}
        edge_index_dict = original_batch.edge_index_dict
        return self.model(x_dict, edge_index_dict, original_batch)


# ---------------------------------------------------------------------------
# Helpers: rank features by attribution; convert payload-byte importances
# to a printable string. Used in notebook plots reproducing paper Fig. 6 / 7.
# ---------------------------------------------------------------------------


def top_flow_features(attr: IGAttribution,
                      feature_names: Sequence[str],
                      top_n: int = 10):
    """Return the top-N flow features by absolute IG attribution as
    [(name, attribution, actual_value), ...] sorted descending."""
    importance = np.abs(attr.flow_attr)
    if len(feature_names) != importance.shape[0]:
        raise ValueError(
            f"feature_names has {len(feature_names)} entries but flow_attr "
            f"has shape {importance.shape}"
        )
    order = np.argsort(-importance)[:top_n]
    return [(feature_names[i], float(attr.flow_attr[i]), float(attr.flow_values[i]))
            for i in order]


def top_packet_features(attr: IGAttribution,
                        feature_names: Sequence[str],
                        top_n: int = 10,
                        aggregate: str = "mean"):
    """Aggregate per-packet IG attributions across packets and return the
    top-N packet features. ``feature_names`` must enumerate the *protocol-
    level* part of the packet attribute (i.e. the leading ``len(feature_names)``
    columns -- typically the 14 protocol features defined in Functions.py).
    """
    if attr.packet_attr.size == 0:
        return []
    n_named = len(feature_names)
    proto_attr = np.abs(attr.packet_attr[:, :n_named])
    if aggregate == "mean":
        agg = proto_attr.mean(axis=0)
    elif aggregate == "sum":
        agg = proto_attr.sum(axis=0)
    elif aggregate == "max":
        agg = proto_attr.max(axis=0)
    else:
        raise ValueError(f"unknown aggregate: {aggregate}")
    order = np.argsort(-agg)[:top_n]
    avg_value = attr.packet_values[:, :n_named].mean(axis=0)
    return [(feature_names[i], float(agg[i]), float(avg_value[i])) for i in order]


def top_payload_bytes(attr: IGAttribution,
                      proto_feat_count: int = 14,
                      top_n: int = 32) -> bytes:
    """Aggregate the per-packet payload-byte attributions, then return the
    top_n bytes as raw bytes. Payload bytes are assumed to start at column
    ``proto_feat_count`` of the packet attribute tensor (matching the layout
    produced by ``NIDSDataset._get_packet_node_features``).
    """
    if attr.packet_attr.size == 0:
        return b""
    payload_attr = np.abs(attr.packet_attr[:, proto_feat_count:])
    if payload_attr.size == 0:
        return b""
    norms = np.linalg.norm(payload_attr, axis=1, keepdims=True) + 1e-12
    payload_attr = payload_attr / norms
    avg = payload_attr.mean(axis=0)
    order = np.argsort(-avg)[:top_n]
    payload_values = attr.packet_values[:, proto_feat_count:].mean(axis=0)[order]
    return bytes(int(round(v)) & 0xFF for v in payload_values)
