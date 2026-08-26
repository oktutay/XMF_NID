#!/bin/bash
# One-shot XMF-GNN pipeline: activates conda env and runs the full pipeline.
#
# Usage:
#   ./run_all.sh                                          # full pipeline
#   ./run_all.sh --skip-extract                           # skip NFStream (CSVs exist)
#   ./run_all.sh --no-ablation                            # skip ablation study
#   ./run_all.sh --epochs 50                              # override epoch count
#   ./run_all.sh --early-stop 10                          # stop if no val_acc gain for 10 epochs
#   ./run_all.sh --skip-extract --skip-features --skip-balance --skip-graphs   # train only
#
# Dataset path: ~/Tutay/Tutay_Sec/CIC_IoT_2023_PCAP/  (shared across projects)
# All other outputs (CSVs, graphs, checkpoints, logs) go under ./data/.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# 1) Activate conda env "xmfgnn"
if ! command -v conda >/dev/null 2>&1; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate xmfgnn

# 2) Sanity check: dependencies
python -c "import torch, torch_geometric, nfstream, captum" 2>/dev/null || {
    echo "[run_all.sh] Some Python dependencies are missing. Run:"
    echo "  pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121"
    echo "  pip install torch_geometric==2.5.3 nfstream==6.5.3 captum==0.7.0"
    exit 1
}

# 3) Run the unified pipeline
python run_pipeline.py "$@"
