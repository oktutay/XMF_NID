"""End-to-end XMF-GNN pipeline (PCAP -> CSV -> graphs -> trained model).

Usage:
    python run_pipeline.py                        # full pipeline
    python run_pipeline.py --skip-extract         # skip NFStream
    python run_pipeline.py --skip-extract --skip-features  # only graph + train
    python run_pipeline.py --no-ablation          # skip ablation study
    python run_pipeline.py --epochs 50            # override epoch count

Each step is idempotent: if the output of a step already exists, it is
skipped. Logs go to ./pipeline.log and stdout.
"""
from __future__ import annotations
import argparse
import glob
import logging
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

# --------------------------------------------------------------------------
# Paths -- PCAPs in shared location, project-specific outputs under XMF_GNN
# --------------------------------------------------------------------------
PCAP_DIR        = Path("~/Tutay/Tutay_Sec/CIC_IoT_2023_PCAP").expanduser()
PCAP_ZIP        = PCAP_DIR / "PCAP.zip"
TMP_EXTRACT_DIR = PCAP_DIR / "_tmp_extract"
PROJECT_ROOT    = Path(__file__).resolve().parent
DATA_DIR        = (PROJECT_ROOT / "data").resolve()
CSV_DIR         = DATA_DIR / "Extracted_Flow_Features"
MIN_FREE_GB     = 10  # abort if free disk drops below this during extraction
PROCESSED_ROOT  = DATA_DIR / "processed_xmfgnn"
TRAIN_ROOT      = PROCESSED_ROOT / "train"
TEST_ROOT       = PROCESSED_ROOT / "test"
TRAIN_CSV_NAME  = "df_class_8_train.csv"
TEST_CSV_NAME   = "df_class_8_test.csv"
CHECKPOINT_DIR  = DATA_DIR / "checkpoints"

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
LOG_FILE = PROJECT_ROOT / "pipeline.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("xmfgnn")


def _section(title: str) -> None:
    log.info("=" * 70)
    log.info(title)
    log.info("=" * 70)


# --------------------------------------------------------------------------
# Step 1: NFStream PCAP -> CSV
#
# Two modes:
#   (a) zip mode -- if PCAP.zip exists, extract each .pcap from the zip on
#       demand, run NFStream, then delete the .pcap. Keeps disk overhead
#       bounded by the largest single PCAP (~2 GB for CIC-IoT2023).
#   (b) disk mode -- if PCAPs already sit on disk under PCAP_DIR (no zip),
#       fall back to the original behavior.
# --------------------------------------------------------------------------
def _free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def _extract_one_pcap_from_zip(zf: zipfile.ZipFile,
                                info: zipfile.ZipInfo,
                                dst_root: Path) -> Path:
    """Extract a single pcap entry, return the path written."""
    zf.extract(info, dst_root)
    return dst_root / info.filename


def _run_nfstream(extractor: Path, pcap_path: Path) -> None:
    subprocess.check_call(
        [sys.executable, str(extractor), str(pcap_path), str(CSV_DIR) + os.sep]
    )


def step_extract(force: bool = False, keep_pcap: bool = False,
                 zip_path: Path = PCAP_ZIP) -> None:
    _section("Step 1/6 -- NFStream PCAP -> CSV")
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    extractor = PROJECT_ROOT / "Utility" / "Feature_extractor_flow_packet_combined.py"

    if zip_path.exists():
        _extract_from_zip(extractor, zip_path, force=force, keep_pcap=keep_pcap)
    else:
        _extract_from_disk(extractor, force=force)


def _extract_from_disk(extractor: Path, force: bool) -> None:
    pcaps = sorted(glob.glob(str(PCAP_DIR / "**" / "*.pcap"), recursive=True))
    if not pcaps:
        log.warning("No PCAP files or PCAP.zip found under %s -- skipping extraction.",
                    PCAP_DIR)
        log.warning("If you have the dataset elsewhere, copy or symlink it to that path.")
        return
    log.info("Disk mode: %d PCAP files. Output dir: %s", len(pcaps), CSV_DIR)
    for pcap in pcaps:
        out_csv = CSV_DIR / (Path(pcap).stem + ".csv")
        if out_csv.exists() and not force:
            log.info("  [skip] %s (CSV already exists)", Path(pcap).name)
            continue
        log.info("  extracting %s ...", Path(pcap).name)
        _run_nfstream(extractor, Path(pcap))


def _extract_from_zip(extractor: Path, zip_path: Path, force: bool,
                      keep_pcap: bool) -> None:
    TMP_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Zip mode: streaming pcaps from %s", zip_path)
    log.info("  tmp extract dir: %s (auto-cleaned per file)", TMP_EXTRACT_DIR)
    log.info("  free disk before start: %.1f GB", _free_gb(TMP_EXTRACT_DIR))

    with zipfile.ZipFile(zip_path) as zf:
        # Sort by uncompressed size ascending -- smallest first, so a broken
        # pipeline surfaces quickly without churning the big DDoS/Mirai files.
        pcap_infos = sorted(
            (i for i in zf.infolist() if i.filename.endswith(".pcap")),
            key=lambda i: i.file_size,
        )
        total = len(pcap_infos)
        log.info("  %d PCAP entries in zip", total)

        done = skipped = 0
        for idx, info in enumerate(pcap_infos, 1):
            pcap_basename = Path(info.filename).stem
            out_csv = CSV_DIR / (pcap_basename + ".csv")
            size_gb = info.file_size / (1024 ** 3)
            if out_csv.exists() and not force:
                skipped += 1
                continue

            free_gb = _free_gb(TMP_EXTRACT_DIR)
            if free_gb < size_gb + MIN_FREE_GB:
                log.error("  [abort] free=%.1f GB < needed=%.1f GB for %s",
                          free_gb, size_gb + MIN_FREE_GB, info.filename)
                log.error("          delete some CSVs or free up disk before retrying.")
                return

            log.info("  [%d/%d] %s (%.2f GB compressed=%.2f GB) free=%.1f GB",
                     idx, total, info.filename, size_gb,
                     info.compress_size / (1024 ** 3), free_gb)

            extracted = _extract_one_pcap_from_zip(zf, info, TMP_EXTRACT_DIR)
            try:
                _run_nfstream(extractor, extracted)
                done += 1
            finally:
                if not keep_pcap and extracted.exists():
                    extracted.unlink()
                    # remove the per-attack subdir if it's now empty
                    try:
                        extracted.parent.rmdir()
                    except OSError:
                        pass

        log.info("  zip extract done: %d processed, %d skipped (already had CSV)",
                 done, skipped)
        # final cleanup of tmp dir if empty
        if not keep_pcap:
            try:
                TMP_EXTRACT_DIR.rmdir()
            except OSError:
                pass


# --------------------------------------------------------------------------
# Step 2: Multi-scale temporal features
# --------------------------------------------------------------------------
def step_features(force: bool = False) -> None:
    _section("Step 2/6 -- Multi-scale temporal features (Algorithm 1)")
    sys.path.insert(0, str(PROJECT_ROOT))
    from Utility.Additional_Features import additional_features, DEFAULT_WINDOW_SIZES_SEC

    csvs = [c for c in glob.glob(str(CSV_DIR / "*.csv"))
            if not c.endswith("_test.csv") and "df_class_8" not in c]
    if not csvs:
        log.warning("No per-class CSVs found in %s -- did extraction run?", CSV_DIR)
        return

    sentinel = CSV_DIR / ".features_done"
    if sentinel.exists() and not force:
        log.info("  [skip] multi-scale block already computed (%s)", sentinel)
        return
    for csv in csvs:
        log.info("  enriching %s ...", Path(csv).name)
        additional_features(csv, window_sizes_sec=DEFAULT_WINDOW_SIZES_SEC)
    sentinel.touch()


# --------------------------------------------------------------------------
# Step 3: Per-class filter + balance + standardize
# --------------------------------------------------------------------------
def step_balance(force: bool = False) -> None:
    _section("Step 3/6 -- Filter (MAC) + balance + standardize (Sec. 4.2)")
    sys.path.insert(0, str(PROJECT_ROOT))
    from Utility.Functions import (
        split_csv, Combining_classes, DEFAULT_CIC_IOT2023_LABELS,
        standardize_flow_features,
    )
    import pandas as pd

    train_csv = CSV_DIR / "train" / TRAIN_CSV_NAME
    test_csv  = CSV_DIR / "train" / TEST_CSV_NAME
    if train_csv.exists() and test_csv.exists() and not force:
        log.info("  [skip] balanced train/test CSVs already exist")
        return

    # 1) split per-class CSVs into train + test
    raw_csvs = [c for c in glob.glob(str(CSV_DIR / "*.csv"))
                if not c.endswith("_test.csv") and "df_class_8" not in c]
    for csv in raw_csvs:
        log.info("  split_csv %s", Path(csv).name)
        split_csv(csv, test_sample=4000, number_in_individual_class=20000)

    # 2) Combining_classes assembles per-class trains under <CSV_DIR>/train/
    Combining_classes(
        directory=str(CSV_DIR) + os.sep,
        classes_list=list(DEFAULT_CIC_IOT2023_LABELS.keys()),
        Number_in_individaul_class=20000,
        Number_of_test_samples=4000,
    )

    # 3) Concat per-class trains and tests, write df_class_8_*.csv
    train_dir = CSV_DIR / "train"
    train_dfs = [pd.read_csv(p) for p in glob.glob(str(train_dir / "*_train.csv"))]
    test_dfs  = [pd.read_csv(p) for p in glob.glob(str(train_dir / "*_test.csv"))]
    if not train_dfs or not test_dfs:
        log.error("  Combining_classes produced no per-class CSVs; aborting.")
        return
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df  = pd.concat(test_dfs, ignore_index=True)

    # 4) Standardize numeric flow columns (paper eq. 14)
    ignore = {"Label"} | {c for c in train_df.columns if c.startswith("udps.")}
    train_df, test_df = standardize_flow_features(train_df, test_df, ignore_cols=ignore)
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    log.info("  wrote %s (%d rows) and %s (%d rows)",
             train_csv, len(train_df), test_csv, len(test_df))


# --------------------------------------------------------------------------
# Step 4: Build PyG HeteroData graphs
# --------------------------------------------------------------------------
def step_build_graphs(force: bool = False) -> None:
    _section("Step 4/6 -- Build HeteroData graphs (NIDSDataset)")
    sys.path.insert(0, str(PROJECT_ROOT))
    from Utility.Functions import NIDSDataset, DEFAULT_CIC_IOT2023_LABELS

    src_train = CSV_DIR / "train" / TRAIN_CSV_NAME
    src_test  = CSV_DIR / "train" / TEST_CSV_NAME
    if not src_train.exists() or not src_test.exists():
        log.error("  Balanced CSVs not found at %s and %s -- run step_balance first.",
                  src_train, src_test)
        return

    # NIDSDataset expects the CSV under <root>/raw/<filename>.
    for root, src in [(TRAIN_ROOT, src_train), (TEST_ROOT, src_test)]:
        (root / "raw").mkdir(parents=True, exist_ok=True)
        dst = root / "raw" / src.name
        if not dst.exists():
            shutil.copy2(src, dst)

    # Skip if already processed
    train_done = list((TRAIN_ROOT / "processed").glob("data_*.pt")) if (TRAIN_ROOT / "processed").exists() else []
    test_done  = list((TEST_ROOT  / "processed").glob("data_test_*.pt")) if (TEST_ROOT / "processed").exists() else []
    if train_done and test_done and not force:
        log.info("  [skip] %d train graphs + %d test graphs already on disk",
                 len(train_done), len(test_done))
        return

    log.info("  building train graphs ...")
    train_set = NIDSDataset(
        root=str(TRAIN_ROOT), label_dict=DEFAULT_CIC_IOT2023_LABELS,
        filename=[TRAIN_CSV_NAME], single_file=True,
    )
    log.info("  building test graphs ...")
    test_set = NIDSDataset(
        root=str(TEST_ROOT), label_dict=DEFAULT_CIC_IOT2023_LABELS,
        filename=[TEST_CSV_NAME], single_file=True, test=True,
    )
    log.info("  built %d train, %d test graphs", len(train_set), len(test_set))


# --------------------------------------------------------------------------
# Step 5: Train + evaluate
# --------------------------------------------------------------------------
def _processed_count(root: Path, test: bool) -> int:
    pat = "data_test_*.pt" if test else "data_*.pt"
    if not (root / "processed").exists():
        return 0
    files = list((root / "processed").glob(pat))
    if not test:
        # exclude data_test_*.pt from train count
        files = [f for f in files if not f.name.startswith("data_test_")]
    return len(files)


def step_train(epochs: int = 100, batch_size: int = 64,
               early_stopping_patience: Optional[int] = None) -> None:
    _section(f"Step 5/6 -- Train XMF-GNN (epochs={epochs}, batch_size={batch_size})")

    n_train = _processed_count(TRAIN_ROOT, test=False)
    n_test  = _processed_count(TEST_ROOT,  test=True)
    if n_train == 0 or n_test == 0:
        log.error("  Cannot train: no processed graphs found.")
        log.error("    train graphs in %s/processed: %d", TRAIN_ROOT, n_train)
        log.error("    test  graphs in %s/processed: %d", TEST_ROOT,  n_test)
        log.error("  Make sure the prior steps ran successfully:")
        log.error("    1. PCAPs in %s", PCAP_DIR)
        log.error("    2. Run ./run_all.sh (or ./run_all.sh --skip-extract if CSVs already exist)")
        return

    sys.path.insert(0, str(PROJECT_ROOT))
    import torch
    from torch_geometric.loader import DataLoader
    from Utility import (
        XMFGNN, NIDSDataset, train, test_cm,
        DEFAULT_CIC_IOT2023_LABELS,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("  device: %s", device)

    train_set = NIDSDataset(
        root=str(TRAIN_ROOT), label_dict=DEFAULT_CIC_IOT2023_LABELS,
        filename=[TRAIN_CSV_NAME], single_file=True, skip_processing=True,
    )
    test_set = NIDSDataset(
        root=str(TEST_ROOT), label_dict=DEFAULT_CIC_IOT2023_LABELS,
        filename=[TEST_CSV_NAME], single_file=True, skip_processing=True, test=True,
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size)
    log.info("  train graphs: %d, test graphs: %d", len(train_set), len(test_set))

    sample = train_set[0]
    model = XMFGNN(
        hetero_metadata=sample.metadata(),
        hidden_size=64, attn_size=32,
        num_classes=len(DEFAULT_CIC_IOT2023_LABELS),
        fusion="attn",
    ).to(device)

    args = {
        "epochs": epochs, "lr": 1e-2, "min_lr": 1e-5,
        "scheduler_patience": 5, "scheduler_threshold": 0.01,
        "batch_size": batch_size,
        "early_stopping_patience": early_stopping_patience,
        "best_checkpoint_path": str(CHECKPOINT_DIR / "xmfgnn_best.pt"),
    }
    t0 = time.time()
    history = train(train_loader, model, args, device=device, val_loader=test_loader)
    train_t = time.time() - t0
    log.info("  training done in %.1f s", train_t)

    # Save checkpoint
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / "xmfgnn_attn.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "history": history,
        "args": args,
    }, ckpt_path)
    log.info("  checkpoint saved to %s", ckpt_path)

    log.info("  Final test evaluation:")
    acc, _, _ = test_cm(test_loader, model, device=device)
    log.info("  test accuracy: %.4f", acc)


# --------------------------------------------------------------------------
# Step 6: Ablation (Table 8)
# --------------------------------------------------------------------------
def step_ablation(epochs: int = 100, batch_size: int = 64,
                  early_stopping_patience: Optional[int] = None) -> None:
    _section(f"Step 6/6 -- Ablation (Table 8, epochs={epochs})")

    n_train = _processed_count(TRAIN_ROOT, test=False)
    n_test  = _processed_count(TEST_ROOT,  test=True)
    if n_train == 0 or n_test == 0:
        log.error("  Cannot run ablation: no processed graphs found "
                  "(train=%d, test=%d). Skipping.", n_train, n_test)
        return

    sys.path.insert(0, str(PROJECT_ROOT))
    import torch
    from torch_geometric.loader import DataLoader
    from Utility import XMFGNN, NIDSDataset, train, test, DEFAULT_CIC_IOT2023_LABELS
    import pandas as pd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = NIDSDataset(
        root=str(TRAIN_ROOT), label_dict=DEFAULT_CIC_IOT2023_LABELS,
        filename=[TRAIN_CSV_NAME], single_file=True, skip_processing=True,
    )
    test_set = NIDSDataset(
        root=str(TEST_ROOT), label_dict=DEFAULT_CIC_IOT2023_LABELS,
        filename=[TEST_CSV_NAME], single_file=True, skip_processing=True, test=True,
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size)
    sample = train_set[0]

    VARIANTS = [
        ("B-GNN",   "baseline"),
        ("FM-GNN",  "mlp"),
        ("SA-GNN",  "simple_attn"),
        ("GA-GNN",  "gated"),
        ("MA-GNN",  "multi_head"),
        ("XMF-GNN", "attn"),
    ]
    args = {
        "epochs": epochs, "lr": 1e-2, "min_lr": 1e-5,
        "scheduler_patience": 5, "scheduler_threshold": 0.01,
        "batch_size": batch_size,
        "early_stopping_patience": early_stopping_patience,
    }
    results = {}
    for tag, fusion in VARIANTS:
        log.info("--- %s (fusion=%s) ---", tag, fusion)
        m = XMFGNN(
            hetero_metadata=sample.metadata(),
            hidden_size=64, attn_size=32,
            num_classes=len(DEFAULT_CIC_IOT2023_LABELS),
            fusion=fusion,
        ).to(device)
        t0 = time.time()
        train(train_loader, m, args, device=device, val_loader=test_loader, log_every=10)
        train_t = time.time() - t0
        t0 = time.time()
        acc = test(test_loader, m, device=device)
        test_t = time.time() - t0
        results[tag] = {"accuracy": acc, "train_s": train_t, "test_s": test_t}

        ckpt = CHECKPOINT_DIR / f"ablation_{fusion}.pt"
        torch.save(m.state_dict(), ckpt)

    table = pd.DataFrame(results).T
    log.info("\n%s", table.to_string())
    table.to_csv(CHECKPOINT_DIR / "ablation_results.csv")


# --------------------------------------------------------------------------
# Step 7: Leave-one-attack-out (zero-day) evaluation
# --------------------------------------------------------------------------
def step_zero_day(held_out: str, epochs: int = 100, batch_size: int = 64) -> None:
    """Train on K-1 known classes, evaluate detection on the held-out class.

    Reports per-epoch standard metrics on the known classes and a final
    block of zero-day metrics (FAR on benign, DR on the held-out attack,
    mean attack-score, AUC, TPR@FPR=1%, EER) via ``loao_evaluate``.
    """
    _section(f"Step 7 -- Zero-day (LOAO) eval, held-out class = {held_out!r}")
    n_train = _processed_count(TRAIN_ROOT, test=False)
    n_test  = _processed_count(TEST_ROOT,  test=True)
    if n_train == 0 or n_test == 0:
        log.error("  Cannot run LOAO: processed graphs missing (train=%d, test=%d).",
                  n_train, n_test)
        return

    sys.path.insert(0, str(PROJECT_ROOT))
    import torch
    from torch_geometric.loader import DataLoader
    from Utility import (
        XMFGNN, NIDSDataset, train, loao_evaluate, make_loao_subsets,
        DEFAULT_CIC_IOT2023_LABELS,
    )

    label_dict = DEFAULT_CIC_IOT2023_LABELS
    if held_out not in label_dict:
        log.error("  held_out %r not in label_dict (%s)", held_out, list(label_dict))
        return
    held_id = label_dict[held_out]
    benign_id = label_dict["Benign"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("  device=%s  held_out_id=%d  benign_id=%d", device, held_id, benign_id)

    train_set = NIDSDataset(
        root=str(TRAIN_ROOT), label_dict=label_dict,
        filename=[TRAIN_CSV_NAME], single_file=True, skip_processing=True,
    )
    test_set = NIDSDataset(
        root=str(TEST_ROOT), label_dict=label_dict,
        filename=[TEST_CSV_NAME], single_file=True, skip_processing=True, test=True,
    )

    cache_dir = str(DATA_DIR / "loao_cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    train_known, test_known, test_unseen = make_loao_subsets(
        train_set, test_set, held_out_label_id=held_id, cache_dir=cache_dir,
    )
    log.info("  train_known=%d  test_known=%d  test_unseen=%d",
             len(train_known), len(test_known), len(test_unseen))
    if len(test_unseen) == 0:
        log.error("  No held-out samples in test set; cannot evaluate zero-day.")
        return

    train_loader  = DataLoader(train_known, batch_size=batch_size, shuffle=True)
    known_loader  = DataLoader(test_known,  batch_size=batch_size)
    unseen_loader = DataLoader(test_unseen, batch_size=batch_size)

    sample = train_set[0]
    model = XMFGNN(
        hetero_metadata=sample.metadata(),
        hidden_size=64, attn_size=32,
        num_classes=len(label_dict),  # keep K outputs; held-out class never seen in training
        fusion="attn",
    ).to(device)
    args = {
        "epochs": epochs, "lr": 1e-2, "min_lr": 1e-5,
        "scheduler_patience": 5, "scheduler_threshold": 0.01,
        "batch_size": batch_size,
    }
    log.info("  training on %d K-1 graphs ...", len(train_known))
    train(train_loader, model, args, device=device, val_loader=known_loader)

    log.info("  evaluating zero-day detection ...")
    metrics = loao_evaluate(model, known_loader, unseen_loader,
                            benign_label=benign_id, device=device)

    ckpt = CHECKPOINT_DIR / f"loao_{held_out}.pt"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "held_out": held_out, "metrics": metrics, "args": args}, ckpt)
    log.info("  checkpoint saved to %s", ckpt)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="XMF-GNN end-to-end pipeline")
    p.add_argument("--skip-extract",  action="store_true", help="skip NFStream PCAP -> CSV")
    p.add_argument("--skip-features", action="store_true", help="skip multi-scale features")
    p.add_argument("--skip-balance",  action="store_true", help="skip filter/balance")
    p.add_argument("--skip-graphs",   action="store_true", help="skip graph build")
    p.add_argument("--skip-train",    action="store_true", help="skip training step")
    p.add_argument("--no-ablation",   action="store_true", help="skip ablation study")
    p.add_argument("--force",         action="store_true", help="re-run even if outputs exist")
    p.add_argument("--epochs",        type=int, default=100)
    p.add_argument("--batch-size",    type=int, default=64)
    p.add_argument("--early-stop",    type=int, default=None, metavar="PATIENCE",
                   help="early stopping: stop if val_acc does not improve for PATIENCE epochs "
                        "(e.g. --early-stop 10). Default: disabled.")
    p.add_argument("--zip-path",      type=Path, default=PCAP_ZIP,
                   help="path to PCAP.zip (zip-mode extract). If absent, falls back to disk mode.")
    p.add_argument("--keep-pcap",     action="store_true",
                   help="zip mode: keep extracted .pcap files instead of deleting after NFStream.")
    p.add_argument("--zero-day",      type=str, default=None, metavar="CLASS",
                   help="run leave-one-attack-out: hold out CLASS during training "
                        "and report FAR/DR/AUC against it (e.g. Mirai, DDos, Recon).")
    p.add_argument("--zero-day-all",  action="store_true",
                   help="run LOAO for every non-benign class (long).")
    args = p.parse_args()

    log.info("PCAP_DIR        = %s", PCAP_DIR)
    log.info("PCAP_ZIP        = %s (exists=%s)", args.zip_path, args.zip_path.exists())
    log.info("CSV_DIR         = %s", CSV_DIR)
    log.info("PROCESSED_ROOT  = %s", PROCESSED_ROOT)
    log.info("CHECKPOINT_DIR  = %s", CHECKPOINT_DIR)

    t_start = time.time()
    if not args.skip_extract:
        step_extract(force=args.force, keep_pcap=args.keep_pcap, zip_path=args.zip_path)
    if not args.skip_features:
        step_features(force=args.force)
    if not args.skip_balance:
        step_balance(force=args.force)
    if not args.skip_graphs:
        step_build_graphs(force=args.force)
    if not args.skip_train:
        step_train(epochs=args.epochs, batch_size=args.batch_size,
                   early_stopping_patience=args.early_stop)
    if not args.no_ablation:
        step_ablation(epochs=args.epochs, batch_size=args.batch_size,
                      early_stopping_patience=args.early_stop)
    if args.zero_day:
        step_zero_day(args.zero_day, epochs=args.epochs, batch_size=args.batch_size)
    if args.zero_day_all:
        sys.path.insert(0, str(PROJECT_ROOT))
        from Utility import DEFAULT_CIC_IOT2023_LABELS
        for cls in DEFAULT_CIC_IOT2023_LABELS:
            if cls == "Benign":
                continue
            step_zero_day(cls, epochs=args.epochs, batch_size=args.batch_size)
    log.info("PIPELINE FINISHED in %.1f s", time.time() - t_start)


if __name__ == "__main__":
    main()
