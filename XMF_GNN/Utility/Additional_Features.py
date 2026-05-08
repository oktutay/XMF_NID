"""
Multi-scale explainable feature extractor for XMF-GNN.

Implements paper Section 3.1 / Algorithm 1: a sliding-window temporal
feature extractor evaluated *in parallel at multiple window sizes*. The
paper enumerates a set W = {w_1, ..., w_n} of window sizes (mentioned
explicitly: 10 s, 30 s, 60 s, 300 s) and concatenates the per-window
feature blocks into the final flow node attribute, written in eq. (just
above Sec. 3.2):

    "FlowFeature" = [F_{10s}, F_{30s}, F_{60s}, F_{300s}, ...] in R^n

Paper Section 3.2 specifies that flow node features are:
  - 76 base flow-level features (NFStream output) +
  - 52 multi-scale temporal features = 128-dim total.

A 52-dim block split evenly across 4 windows means **13 features per
window**. We follow the example list in paper Algorithm 1 lines 12-16
("Packet rate (pkt/s); Average packet size; TCP SYN ratio; Directional
imbalance (src/dst ratio); ICMP proportion, RTT variance, etc."). The
13 chosen features below enumerate the Algorithm 1 examples plus the
additional behavioral statistics implied by "etc." -- byte rate, payload
ratio, IAT statistics, port-spread, vulnerable-port hits.

Per-window features (13 each, x 4 windows = 52):
    1.  packet_rate          packets per second
    2.  byte_rate            bytes per second
    3.  avg_pkt_size         mean packet size in window
    4.  syn_ratio            TCP SYN packets / total packets
    5.  ack_ratio            TCP ACK packets / total packets
    6.  fin_rst_ratio        (FIN+RST) / total packets
    7.  icmp_ratio           ICMP packets / total packets
    8.  udp_ratio            UDP packets / total packets
    9.  src_dst_ratio        directional imbalance (src->dst / dst->src)
    10. iat_mean             mean inter-arrival time in window
    11. iat_std              std of IAT in window
    12. unique_dst_ports     unique destination ports observed
    13. vuln_port_hits       hits on the well-known vulnerable port list

Each row of the input CSV (one *flow*) is annotated with these 13 stats
evaluated over the four windows (10/30/60/300 s) ending at the flow's
``bidirectional_first_seen_ms`` timestamp, computed against all *prior*
flows that share the same destination IP -- exactly the per-destination
buffer described in Algorithm 1 lines 2-4 ("for each destination D_j ...
Initialize buffer W_j[w]").

The paper does not pin which side of the bidirectional flow counts as the
"sender", so we use ``src2dst_*`` / ``dst2src_*`` NFStream columns and
treat src->dst as the dominant direction. SYN/ACK/FIN/RST counts come
from NFStream's ``bidirectional_*_packets`` columns.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


# Window sizes in *seconds* (paper Sec. 3.1, "10 s, 30 s, 60 s, 300 s ...").
DEFAULT_WINDOW_SIZES_SEC = (10, 30, 60, 300)

# Per-window feature names. The order MUST stay stable across runs because the
# trained model relies on it for IG attribution (Sec. 4.6.4) and notebook
# plots.
PER_WINDOW_FEATURE_NAMES: Tuple[str, ...] = (
    "pkt_rate",
    "byte_rate",
    "avg_pkt_size",
    "syn_ratio",
    "ack_ratio",
    "fin_rst_ratio",
    "icmp_ratio",
    "udp_ratio",
    "src_dst_ratio",
    "iat_mean",
    "iat_std",
    "unique_dst_ports",
    "vuln_port_hits",
)
N_PER_WINDOW = len(PER_WINDOW_FEATURE_NAMES)  # = 13

# Vulnerable / sensitive ports for feature 13 (matches the list used by the
# XG-NID feature extractor; see GNN4ID Additional_Features.py).
DEFAULT_VULNERABLE_PORTS = frozenset(
    {20, 21, 22, 23, 25, 53, 80, 110, 143, 443, 445, 3389, 8080}
)


def window_feature_names(window_sizes_sec: Iterable[int] = DEFAULT_WINDOW_SIZES_SEC
                         ) -> List[str]:
    """Return the full ordered list of multi-scale feature column names."""
    out: List[str] = []
    for w in window_sizes_sec:
        suffix = f"_{int(w)}s"
        out.extend([f + suffix for f in PER_WINDOW_FEATURE_NAMES])
    return out


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Element-wise division that returns 0 where ``den`` is zero."""
    out = np.zeros_like(num, dtype=np.float64)
    mask = den != 0
    out[mask] = num[mask] / den[mask]
    return out


def _compute_per_destination_window(df: pd.DataFrame,
                                    window_sec: int,
                                    vuln_ports: frozenset) -> np.ndarray:
    """Compute the 13 temporal features for every row of ``df`` over a
    sliding window of ``window_sec`` seconds, grouped by destination IP.

    Returns a (N, 13) float64 array aligned to ``df.index``.
    """
    # The CSV lists times in ms (NFStream's bidirectional_first_seen_ms).
    t_ms = df["bidirectional_first_seen_ms"].astype(np.int64)
    win_ms = int(window_sec) * 1000

    # We need the *prior* flows in the window per destination. Sort by time
    # (already done by caller) and walk per-destination using a deque of
    # indices whose flow timestamp is within the window.
    n = len(df)
    feats = np.zeros((n, N_PER_WINDOW), dtype=np.float64)

    # Pre-extract numpy arrays for speed (avoids row-wise pandas access).
    dst_ip = df["dst_ip"].astype(str).to_numpy()
    pkts = df["bidirectional_packets"].to_numpy(dtype=np.float64)
    bytes_ = df["bidirectional_bytes"].to_numpy(dtype=np.float64)
    duration_ms = df["bidirectional_duration_ms"].to_numpy(dtype=np.float64)
    syn = df["bidirectional_syn_packets"].to_numpy(dtype=np.float64)
    ack = df["bidirectional_ack_packets"].to_numpy(dtype=np.float64)
    fin = df["bidirectional_fin_packets"].to_numpy(dtype=np.float64)
    rst = df["bidirectional_rst_packets"].to_numpy(dtype=np.float64)
    proto = df["protocol"].to_numpy(dtype=np.int64)
    src2dst_pkts = df["src2dst_packets"].to_numpy(dtype=np.float64)
    dst2src_pkts = df["dst2src_packets"].to_numpy(dtype=np.float64)
    src2dst_mean_ps = df.get("src2dst_mean_ps", pd.Series(np.zeros(n))).to_numpy(dtype=np.float64)
    dst2src_mean_ps = df.get("dst2src_mean_ps", pd.Series(np.zeros(n))).to_numpy(dtype=np.float64)
    src2dst_min_piat_ms = df.get("src2dst_min_piat_ms", pd.Series(np.zeros(n))).to_numpy(dtype=np.float64)
    src2dst_max_piat_ms = df.get("src2dst_max_piat_ms", pd.Series(np.zeros(n))).to_numpy(dtype=np.float64)
    src2dst_mean_piat_ms = df.get("src2dst_mean_piat_ms", pd.Series(np.zeros(n))).to_numpy(dtype=np.float64)
    src2dst_stddev_piat_ms = df.get("src2dst_stddev_piat_ms", pd.Series(np.zeros(n))).to_numpy(dtype=np.float64)
    dst_port = df["dst_port"].to_numpy(dtype=np.int64)
    t_arr = t_ms.to_numpy()

    # Pre-group row indices by destination, in time order (df was sorted).
    from collections import defaultdict, deque
    groups: dict = defaultdict(list)
    for i, d in enumerate(dst_ip):
        groups[d].append(i)

    for dst, idx_list in groups.items():
        idx_arr = np.array(idx_list, dtype=np.int64)
        # Deque of (i, t) for currently-in-window prior flows.
        window: deque = deque()
        for cur_pos in range(len(idx_arr)):
            i = idx_arr[cur_pos]
            t_i = t_arr[i]
            # Pop entries that fell out of the [t_i - win_ms, t_i] window.
            while window and (t_i - t_arr[window[0]]) > win_ms:
                window.popleft()

            if not window:
                # No prior context -> features stay at 0.0. Add the current
                # flow into the window for future iterations and continue.
                window.append(i)
                continue

            members = np.fromiter(window, dtype=np.int64)

            tot_pkts = pkts[members].sum()
            tot_bytes = bytes_[members].sum()
            tot_dur_ms = duration_ms[members].sum()
            # Window length used for *_rate (paper says "pkt/s"). Use the
            # actual span between the oldest in-window flow and i (in s).
            span_sec = max((t_arr[i] - t_arr[members[0]]) / 1000.0, 1e-3)

            tot_syn = syn[members].sum()
            tot_ack = ack[members].sum()
            tot_fin_rst = fin[members].sum() + rst[members].sum()
            tot_icmp = ((proto[members] == 1).astype(np.float64) * pkts[members]).sum()
            tot_udp = ((proto[members] == 17).astype(np.float64) * pkts[members]).sum()

            tot_src_pkts = src2dst_pkts[members].sum()
            tot_dst_pkts = dst2src_pkts[members].sum()

            avg_pkt_size = float(np.mean(
                np.concatenate([src2dst_mean_ps[members], dst2src_mean_ps[members]])
            )) if tot_pkts > 0 else 0.0

            # IAT mean / std taken from src2dst flow stats averaged across
            # the in-window flows.
            iat_mean = float(src2dst_mean_piat_ms[members].mean())
            iat_std = float(src2dst_stddev_piat_ms[members].mean())

            unique_dst_ports = float(len(set(dst_port[members].tolist())))
            vuln_port_hits = float(np.sum(np.isin(dst_port[members], list(vuln_ports))))

            feats[i, 0] = tot_pkts / span_sec                # pkt_rate
            feats[i, 1] = tot_bytes / span_sec               # byte_rate
            feats[i, 2] = avg_pkt_size                       # avg_pkt_size
            feats[i, 3] = float(_safe_div(np.array([tot_syn]), np.array([tot_pkts]))[0])
            feats[i, 4] = float(_safe_div(np.array([tot_ack]), np.array([tot_pkts]))[0])
            feats[i, 5] = float(_safe_div(np.array([tot_fin_rst]), np.array([tot_pkts]))[0])
            feats[i, 6] = float(_safe_div(np.array([tot_icmp]), np.array([tot_pkts]))[0])
            feats[i, 7] = float(_safe_div(np.array([tot_udp]), np.array([tot_pkts]))[0])
            feats[i, 8] = float(_safe_div(np.array([tot_src_pkts]), np.array([tot_dst_pkts]))[0])
            feats[i, 9] = iat_mean
            feats[i, 10] = iat_std
            feats[i, 11] = unique_dst_ports
            feats[i, 12] = vuln_port_hits

            window.append(i)

    return feats


def additional_features(file_name: str,
                        window_sizes_sec: Iterable[int] = DEFAULT_WINDOW_SIZES_SEC,
                        vulnerable_ports: Iterable[int] = DEFAULT_VULNERABLE_PORTS,
                        exp_id: Tuple[int, ...] = (0, -1),
                        proto_list: Tuple[int, ...] = (1, 2, 6, 17, 58)
                        ) -> str:
    """Add 4 x 13 = 52 multi-scale temporal features to the NFStream CSV.

    The function reads ``file_name``, sorts by ``bidirectional_first_seen_ms``,
    computes per-destination rolling windows for each window size in
    ``window_sizes_sec``, concatenates the resulting blocks, one-hot encodes
    ``expiration_id`` and ``protocol``, and overwrites the CSV in place.

    Returns ``file_name`` on success, an empty string on read failure.
    """
    vuln_ports = frozenset(vulnerable_ports)

    try:
        data = pd.read_csv(file_name)
    except Exception:
        print(f"file reading error: {file_name}")
        return ""

    if "bidirectional_first_seen_ms" not in data.columns:
        print(f"CSV missing bidirectional_first_seen_ms: {file_name}")
        return ""

    data = data.sort_values(by="bidirectional_first_seen_ms").reset_index(drop=True)

    # Compute per-window blocks and assign columns.
    for w in window_sizes_sec:
        block = _compute_per_destination_window(data, int(w), vuln_ports)
        for j, base_name in enumerate(PER_WINDOW_FEATURE_NAMES):
            data[f"{base_name}_{int(w)}s"] = block[:, j]

    # One-hot encode the categorical NFStream fields (kept consistent with
    # the XG-NID code-base so downstream NIDSDataset.process() works).
    data["expiration_id"] = pd.Categorical(data["expiration_id"], categories=list(exp_id))
    data["protocol"] = pd.Categorical(data["protocol"], categories=list(proto_list))
    data = pd.get_dummies(
        data, prefix=["Exp", "proto"], columns=["expiration_id", "protocol"], dtype=int
    )

    data.to_csv(file_name, index=False)
    return file_name
