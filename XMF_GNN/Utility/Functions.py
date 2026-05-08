"""
Dataset, preprocessing, and utility helpers for XMF-GNN.

Differences vs. the XG-NID GNN4ID Functions.py
----------------------------------------------
Paper Section 3.2 ("Graph-level construction") of XMF-GNN explicitly removes
the explicit ``contain`` and ``link`` edge types kept by XG-NID. Quote:

    "we eliminate the explicit representation of both edge types and instead
     retain only the connection relationships between the two types of
     nodes. The original edge features are embedded directly into the
     corresponding node attributes."

Implementation:
- The ``contain`` edge attributes (packet direction, IP size, transport
  size, payload size -- all per-packet) are folded into the corresponding
  *packet node* attribute -- they describe per-packet metadata anyway.
- The ``link`` edge attribute (delta_time between consecutive packets)
  is also stored on the packet node (delta_time of the packet relative to
  the previous packet within the flow); zero for the first packet.
- A single edge type ``('flow','connected_to','packet')`` is emitted, and
  ``T.ToUndirected()`` adds the reverse so message passing flows both ways
  between flow and packet (eq. 3 of paper Sec. 3.2 keeps E_{f,p} symbolic;
  ToUndirected is the standard PyG idiom for it).

Therefore the per-packet node feature vector is:

    [direction, ip_size, transport_size, payload_size, delta_time]   -- 5 dims
        ++ [14 protocol-header flag features (syn, cwr, ece, urg, ack, psh,
            rst, fin, + 6 numeric: ip_size_norm, payload_size_norm,
            transport_size_norm, ttl, ip_pkt_len, mss)]              -- but paper
                                                                        says "14
                                                                        packet-level
                                                                        features
                                                                        with an
                                                                        emphasis
                                                                        on
                                                                        protocol
                                                                        fields".
        ++ [1500-dim payload byte vector]                            -- 1500 dims

Paper Sec. 3.2: "comprising a 1500-dimensional payload encoding and 14
protocol-level header features." We therefore use exactly **14 protocol /
edge-derived features** stacked with the 1500 payload bytes -> 1514-dim
packet node feature.

The 14 packet-level features we use are the 8 TCP flag bits available in
NFStream's ``udps`` plugin (syn, cwr, ece, urg, ack, psh, rst, fin) plus
the 6 numeric per-packet quantities that originally lived on the edges:
``packet_direction``, ``ip_size``, ``transport_size``, ``payload_size``,
``delta_time``, ``transport_size / ip_size`` (a small derived ratio that
captures payload density, kept to reach the paper's 14-feature count).
"""

from __future__ import annotations

import glob
import os
import random
import re
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Dataset, HeteroData
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants matching paper Section 4.1 (CIC-IoT-2023) and Table 3 settings.
# ---------------------------------------------------------------------------

# Default 8-class CIC-IoT2023 label dict. Paper Tables 1, 2, 4, 6 list these
# attack categories. Aliases for Dos/DDos vs DoS/DDoS are accepted via
# ``_normalize_label_dict`` below.
DEFAULT_CIC_IOT2023_LABELS = {
    "Benign": 0,
    "WebBased": 1,
    "Spoofing": 2,
    "Recon": 3,
    "Mirai": 4,
    "Dos": 5,
    "DDos": 6,
    "BruteForce": 7,
}

# Attacker MAC addresses for CIC-IoT2023 -- paper Sec. 4.2 ("attacker
# identification information"). Same list used by the XG-NID code base.
CIC_IOT2023_ATTACKER_MACS = frozenset({
    "dc:a6:32:dc:27:d5", "e4:5f:01:55:90:c4", "dc:a6:32:c9:e4:ab",
    "ac:17:02:05:34:27", "dc:a6:32:c9:e5:a4", "dc:a6:32:c9:e4:d5",
    "dc:a6:32:c9:e5:ef", "dc:a6:32:c9:e4:90", "b0:09:da:3e:82:6c",
})


def _normalize_label_dict(label_dict: Dict[str, int]) -> Dict[str, int]:
    """Accept both Dos/DDos (code) and DoS/DDoS (paper) spellings."""
    aliases = {"DoS": "Dos", "DDoS": "DDos"}
    out = dict(label_dict)
    for paper_form, code_form in aliases.items():
        if code_form in out and paper_form not in out:
            out[paper_form] = out[code_form]
    return out


# ---------------------------------------------------------------------------
# NIDSDataset: read NFStream CSVs, emit HeteroData graphs.
# ---------------------------------------------------------------------------


class NIDSDataset(Dataset):
    """Graph dataset for XMF-GNN.

    Builds one ``HeteroData`` per row of the input CSV. Each row is a flow,
    its associated 1..20 packets, and a 4*13=52-dim multi-scale temporal
    feature block computed offline by ``Additional_Features.additional_features``.

    Args:
        root: root directory; PyG persists graph objects to ``root/processed/``.
        label_dict: mapping ``class_name -> integer label`` (8 classes for
            CIC-IoT2023; 15 for CIC-IDS2017).
        filename: list of CSV file paths to consume.
        index_num: starting index for ``data_{i}.pt`` filenames; useful when
            appending to an existing processed/ directory.
        skip_processing: if True, skip ``process()`` and use existing files.
        include_packetflag: if True, packet node features include the 8 TCP
            flag bits (kept True by default, they are part of the paper's
            14 protocol-level packet features).
        include_packetpayload: include the 1500-dim payload byte vector.
        test: write data_test_{i}.pt instead of data_{i}.pt.
        single_file: True if the CSV already has a 'Label' column.
    """

    def __init__(self,
                 root: str,
                 label_dict: Dict[str, int],
                 filename: Sequence[str],
                 index_num: int = 0,
                 skip_processing: bool = False,
                 include_packetflag: bool = True,
                 include_packetpayload: bool = True,
                 test: bool = False,
                 single_file: bool = False,
                 transform=None,
                 pre_transform=None):
        self.test = test
        self.filename = list(filename)
        self.include_packetflag = include_packetflag
        self.include_packetpayload = include_packetpayload
        self.index = index_num
        self.label_dict = _normalize_label_dict(label_dict)
        self.skip_processing = skip_processing
        self.length = 0
        self.single_file = single_file
        super().__init__(root, transform, pre_transform)

    # ------------------------------------------------------------------
    # PyG Dataset API
    # ------------------------------------------------------------------
    @property
    def raw_file_names(self):
        return self.filename

    @property
    def processed_file_names(self):
        if self.skip_processing:
            if self.test:
                lst = glob.glob(os.path.join(self.root, "processed/data_test*"))
                self.length = len(lst)
                return lst[0] if lst else "data_test_0.pt"
            else:
                lst = glob.glob(os.path.join(self.root, "processed/data*"))
                test_lst = glob.glob(os.path.join(self.root, "processed/data_test*"))
                self.length = len(lst) - len(test_lst)
                return lst[0] if lst else "data_0.pt"
        return []

    def download(self):
        pass

    # ------------------------------------------------------------------
    def process(self):
        for files in self.raw_paths:
            self.data = pd.read_csv(files)
            print("Reading File ---> " + os.path.basename(files), file=sys.stderr)

            if not self.single_file:
                # Drop NFStream metadata columns that are not flow-level
                # behavioral features (per paper count of 76 flow features).
                drop_meta = [
                    "src_ip", "src_port", "dst_ip", "dst_port", "ip_version",
                    "id", "src_mac", "src_oui", "dst_mac", "dst_oui",
                    "vlan_id", "tunnel_id",
                    "bidirectional_first_seen_ms", "bidirectional_last_seen_ms",
                    "src2dst_first_seen_ms", "src2dst_last_seen_ms",
                    "dst2src_first_seen_ms", "dst2src_last_seen_ms",
                    "bidirectional_syn_packets", "bidirectional_cwr_packets",
                    "bidirectional_ece_packets", "bidirectional_urg_packets",
                    "bidirectional_ack_packets", "bidirectional_psh_packets",
                    "bidirectional_rst_packets", "bidirectional_fin_packets",
                ]
                self.data.drop(columns=[c for c in drop_meta if c in self.data.columns],
                               inplace=True)
                # Get label from filename and provided dictionary
                label = self._get_labels(files)
            else:
                label = None  # row-level Label column; resolved per row below

            # ------------------------------------------------------------------
            # Convert serialized list columns back to Python lists.
            # NFStream's CSV writer flattens udps lists with str(); we reverse it.
            # ------------------------------------------------------------------
            list_cols = [
                "udps.payload_data",
                "udps.packet_direction",
                "udps.ip_size",
                "udps.transport_size",
                "udps.payload_size",
                "udps.delta_time",
            ]
            if self.include_packetflag:
                list_cols += [
                    "udps.syn", "udps.cwr", "udps.ece", "udps.urg",
                    "udps.ack", "udps.psh", "udps.rst", "udps.fin",
                ]
            for col in list_cols:
                self.data[col] = self.data[col].map(
                    lambda x: x.strip("][").replace("'", "").split(", ")
                )

            for _, flow in tqdm(self.data.iterrows(), total=self.data.shape[0]):
                flow_node_feats = self._get_flow_node_features(flow)
                packet_node_feats = self._get_packet_node_features(flow)
                connected_edge_index = self._get_connected_edge_index(
                    len(flow["udps.payload_data"])
                )

                if self.single_file:
                    label = torch.tensor(np.asarray(flow["Label"]), dtype=torch.int64)

                data = HeteroData()
                data["flow"].x = flow_node_feats
                data["packet"].x = packet_node_feats
                # Paper Sec. 3.2: single 'connected_to' relation; no edge attrs.
                data["flow", "connected_to", "packet"].edge_index = connected_edge_index
                data.y = label

                # ToUndirected adds the reverse edge so packets can also send
                # messages back to flow nodes -- required by PyG message
                # passing for the second SAGE layer.
                data = T.ToUndirected()(data)

                fname = (
                    f"data_test_{self.index}.pt" if self.test
                    else f"data_{self.index}.pt"
                )
                torch.save(data, os.path.join(self.processed_dir, fname))
                self.index += 1

            if not self.skip_processing:
                if self.test:
                    lst = glob.glob(os.path.join(self.root, "processed/data_test*"))
                    self.length = len(lst)
                else:
                    lst = glob.glob(os.path.join(self.root, "processed/data*"))
                    tlst = glob.glob(os.path.join(self.root, "processed/data_test*"))
                    self.length = len(lst) - len(tlst)

    # ------------------------------------------------------------------
    # Per-row feature extractors
    # ------------------------------------------------------------------
    def _get_flow_node_features(self, flow) -> torch.Tensor:
        """Extract the flow-level feature vector.

        Drops the per-packet udps.* lists (which become packet-node features)
        and any optional Label column. The remainder is the flow node attribute:
        ~76 base flow features + 52 multi-scale temporal features = ~128 dims.
        """
        drop_cols = [
            "udps.payload_data", "udps.delta_time", "udps.packet_direction",
            "udps.ip_size", "udps.transport_size", "udps.payload_size",
            "udps.syn", "udps.cwr", "udps.ece", "udps.urg",
            "udps.ack", "udps.psh", "udps.rst", "udps.fin",
        ]
        if self.single_file and "Label" in flow.index:
            drop_cols.append("Label")
        flow_data = flow.drop(labels=[c for c in drop_cols if c in flow.index])

        arr = np.asarray(flow_data, dtype=float).reshape(1, -1)
        return torch.tensor(arr, dtype=torch.float32)

    def _get_packet_node_features(self, flow) -> torch.Tensor:
        """Extract packet-level features: 14 protocol-level features +
        1500-dim payload (paper Sec. 3.2). The 14 features are the original
        ``contain``/``link`` edge attributes folded into the packet node:
            [direction, ip_size, transport_size, payload_size, delta_time,
             syn, cwr, ece, urg, ack, psh, rst, fin, payload_density]

        ``payload_density`` = payload_size / max(ip_size, 1) keeps the count
        at 14 and provides a useful per-packet ratio.
        """
        n_pkts = len(flow["udps.payload_data"])
        feats: List[List[float]] = []

        for i in range(n_pkts):
            row: List[float] = []
            # 5 numeric "edge-folded" features
            ip_size = float(flow["udps.ip_size"][i])
            transport_size = float(flow["udps.transport_size"][i])
            payload_size = float(flow["udps.payload_size"][i])
            row.append(float(flow["udps.packet_direction"][i]))
            row.append(ip_size)
            row.append(transport_size)
            row.append(payload_size)
            row.append(float(flow["udps.delta_time"][i]))

            # 8 TCP flag bits
            if self.include_packetflag:
                for flag in ("syn", "cwr", "ece", "urg", "ack", "psh", "rst", "fin"):
                    row.append(float(flow[f"udps.{flag}"][i]))
            else:
                row.extend([0.0] * 8)

            # 1 derived: payload_density = payload_size / ip_size  (-> 14 total)
            row.append(payload_size / ip_size if ip_size > 0 else 0.0)

            # Payload bytes (1500-dim)
            if self.include_packetpayload:
                hex_str = flow["udps.payload_data"][i]
                if not hex_str or hex_str == "00":
                    payload = np.zeros(1500, dtype=np.int64)
                else:
                    try:
                        b = bytes.fromhex(hex_str)
                    except ValueError:
                        b = b""
                    blist = list(b)
                    if len(blist) < 1500:
                        payload = np.pad(np.asarray(blist, dtype=np.int64),
                                          (0, 1500 - len(blist)), "constant")
                    else:
                        payload = np.asarray(blist[:1500], dtype=np.int64)
                row.extend(payload.tolist())
            feats.append(row)

        arr = np.asarray(feats, dtype=np.float32)
        return torch.tensor(arr, dtype=torch.float32)

    def _get_connected_edge_index(self, n_packets: int) -> torch.Tensor:
        """Return the (2, n_packets) edge index for ``flow -> packet``.

        Paper Sec. 3.2: a single 'connected_to' relation, no edge features.
        Index 0 is the (only) flow node; indices 0..n_packets-1 are packets.
        """
        flow_idx = np.zeros(n_packets, dtype=np.int64)
        packet_idx = np.arange(n_packets, dtype=np.int64)
        edge = np.vstack([flow_idx, packet_idx])
        return torch.tensor(edge, dtype=torch.int64)

    def _get_labels(self, file_name: str) -> torch.Tensor:
        name = os.path.basename(file_name)
        cls = name.split("-")[0]
        if cls not in self.label_dict:
            raise KeyError(
                f"File {name} resolves to class {cls!r} which is not in "
                f"label_dict={list(self.label_dict.keys())}"
            )
        return torch.tensor(np.asarray([self.label_dict[cls]]), dtype=torch.int64)

    # ------------------------------------------------------------------
    def len(self):
        return self.length

    def get(self, idx: int):
        fname = f"data_test_{idx}.pt" if self.test else f"data_{idx}.pt"
        return torch.load(os.path.join(self.processed_dir, fname))


# ---------------------------------------------------------------------------
# CIC-IoT-2023 file utilities (paper Sec. 4.1.2 / 4.2 -- mostly identical to
# the XG-NID code base; kept here so XMF-GNN is self-contained).
# ---------------------------------------------------------------------------


def rename_files(directory: str, name_mapping: Dict[str, str]) -> None:
    """Bulk-rename PCAP files using a class-name mapping (notebook helper)."""
    if not os.path.exists(directory):
        print(f"The directory '{directory}' does not exist.")
        return
    pattern = directory + "\\**\\*pcap"
    for filename in glob.glob(pattern):
        if not os.path.isfile(filename):
            continue
        base = filename.split("\\")[-1]
        old = re.sub(r"\d+", "", base.split(".")[-2])
        old = old[:-1] if old.endswith("_") else old
        new_name_part = name_mapping.get(old, old)
        new_name = os.path.dirname(filename) + "\\" + new_name_part
        try:
            number = re.search(r"\d+", os.path.basename(filename)).group(0)
            new_name = new_name + "_" + number + ".pcap"
        except Exception:
            new_name = new_name + "_0.pcap"
        os.rename(filename, new_name)
        print(f"Renamed: {os.path.basename(filename)} -> {os.path.basename(new_name)}")


def duplicate_rows(df: pd.DataFrame, target_rows: int) -> pd.DataFrame:
    """Deterministically duplicate rows to reach ``target_rows``.

    Paper Sec. 4.2: minority classes are oversampled, majority are
    undersampled. This helper handles the oversampling side.
    """
    if len(df) == 0 or target_rows <= len(df):
        return df.copy()
    original = df.copy()
    iters = (target_rows // df.shape[0]) - 1
    for _ in range(iters):
        df = pd.concat([df, original], ignore_index=True)
    over = target_rows - df.shape[0]
    return random_pick_rows(df, original, over)


def random_pick_rows(df: pd.DataFrame, original: pd.DataFrame, over: int
                     ) -> pd.DataFrame:
    for _ in range(over):
        i = random.randint(0, original.shape[0] - 1)
        df = pd.concat([df, original.iloc[i:i + 1]], ignore_index=True)
    return df


def split_csv(file_path: str,
              test_sample: int = 4000,
              number_in_individual_class: int = 20000,
              attacker_macs: frozenset = CIC_IOT2023_ATTACKER_MACS) -> None:
    """Paper Sec. 4.2 step 1: filter by attacker MAC, then split into
    train CSV (overwrites ``file_path``) and test CSV.

    Bidirectional filter:
      - For Benign rows: drop any flow that touches any attacker MAC.
      - For attack rows: keep only flows where attacker is src OR dst.
    """
    df = pd.read_csv(file_path)
    name_file = file_path.split("\\")[-1].split(".")[-2]
    name_check = name_file.split("-")[0]

    src_is_attacker = df["src_mac"].isin(attacker_macs)
    dst_is_attacker = df["dst_mac"].isin(attacker_macs)

    if name_check == "Benign":
        df = df[~src_is_attacker & ~dst_is_attacker]
    else:
        df = df[src_is_attacker | dst_is_attacker]

    if df.shape[0] > 35000:
        df_test = df.sample(n=test_sample, random_state=42)
        df = df.drop(df_test.index)
        df = df.sample(n=number_in_individual_class, random_state=42)
    else:
        df_test = df.sample(frac=0.2, random_state=42)
        df = df.drop(df_test.index)

    test_dir = os.path.join(os.path.dirname(file_path), "test")
    os.makedirs(test_dir, exist_ok=True)
    df_test.to_csv(os.path.join(test_dir, name_file + "_test.csv"), index=False)
    df.to_csv(file_path, index=False)


def Combining_classes(directory: str,
                      classes_list: Sequence[str],
                      Number_in_individaul_class: int = 20000,
                      Number_of_test_samples: int = 4000,
                      label_dict: Optional[Dict[str, int]] = None) -> None:
    """Paper Sec. 4.2 step 2: collect per-class CSVs into one balanced
    train CSV and one balanced test CSV per class, then add the integer
    'Label' column. Paper Table 2 / 4 numbers are achieved by setting
    ``Number_in_individaul_class=20000`` and the per-class test sizes via
    ``split_csv``.
    """
    if label_dict is None:
        label_dict = DEFAULT_CIC_IOT2023_LABELS
    label_dict = _normalize_label_dict(label_dict)

    for each_class in tqdm(classes_list):
        train_files = glob.glob(directory + each_class + "*")
        df_list = []
        last_name_file = each_class
        for f in train_files:
            df = pd.read_csv(f)
            last_name_file = f.split("\\")[-1].split(".")[-2].split("-")[0]
            df_list.append(df)

        if not df_list:
            print(f"No CSVs for class {each_class}, skipping.")
            continue

        final_df = pd.concat(df_list, ignore_index=True)
        if final_df.shape[0] <= Number_in_individaul_class:
            final_df = duplicate_rows(final_df, Number_in_individaul_class)
        else:
            frac = Number_in_individaul_class / final_df.shape[0]
            final_df = final_df.sample(frac=frac, random_state=42)

        final_df["Label"] = label_dict[last_name_file]
        train_path = os.path.join(os.path.dirname(train_files[0]),
                                   "train", last_name_file + "_train.csv")
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        final_df.to_csv(train_path, index=False)

        # Test data: stitch the per-class test CSVs together.
        test_files = glob.glob(directory + "test\\" + each_class + "*")
        df_list = []
        for f in test_files:
            df = pd.read_csv(f)
            last_name_file = f.split("\\")[-1].split(".")[-2].split("-")[0]
            df_list.append(df)
            os.remove(f)
        if df_list:
            test_df = pd.concat(df_list, ignore_index=True)
            test_df["Label"] = label_dict[last_name_file]
            if test_df.shape[0] > Number_of_test_samples:
                test_df = test_df.sample(n=Number_of_test_samples, random_state=42)
            test_path = os.path.join(os.path.dirname(test_files[0]),
                                     last_name_file + "_test.csv")
            test_df.to_csv(test_path, index=False)


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def standardize_flow_features(train_df: pd.DataFrame,
                              test_df: pd.DataFrame,
                              ignore_cols: Sequence[str] = ()) -> tuple:
    """Apply StandardScaler-style normalization (paper eq. 14):
        x' = (x - mu) / sigma
    Computed on ``train_df`` and applied to both. Non-numeric columns and
    columns in ``ignore_cols`` are left unchanged.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    for col in train_df.columns:
        if col in ignore_cols or train_df[col].dtype.kind not in "fiu":
            continue
        mu = train_df[col].mean()
        sigma = train_df[col].std(ddof=0)
        if sigma == 0 or np.isnan(sigma):
            continue
        train_df[col] = (train_df[col] - mu) / sigma
        if col in test_df.columns:
            test_df[col] = (test_df[col] - mu) / sigma
    return train_df, test_df
