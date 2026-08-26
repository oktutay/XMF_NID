"""Train shallow / tabular baselines on flow-level features.

Drops per-packet (``udps.*``) columns and reads only the flow-side features
emitted by NFStream + the 52-dim multi-scale temporal block from
``Additional_Features``. Each model is evaluated with both the standard
multi-class metrics block and the zero-day binary metrics block
(FAR/DR/AUC/TPR@FPR/EER).

Usage:
    python train_baselines.py
    python train_baselines.py --held-out Mirai     # LOAO: hold a class out of training
    python train_baselines.py --models rf,mlp      # subset of models
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from Utility.Training import calculate_metrics, zero_day_metrics  # noqa: E402
from Utility.Functions import DEFAULT_CIC_IOT2023_LABELS  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
TRAIN_CSV = DATA_DIR / "Extracted_Flow_Features" / "train" / "df_class_8_train.csv"
TEST_CSV  = DATA_DIR / "Extracted_Flow_Features" / "train" / "df_class_8_test.csv"
OUT_DIR   = DATA_DIR / "baseline_results"


def load_flow_features(csv_path: Path):
    """Return (X, y) using only numeric flow-level columns.

    Drops per-packet (``udps.*``) columns -- those carry packet sequences
    and a 1500-dim payload hex blob, which is the XMF-GNN packet branch.
    Baselines see only flow-level statistics + the multi-scale temporal
    features ``Additional_Features`` already wrote into the CSV.
    """
    df = pd.read_csv(csv_path)
    y = df["Label"].astype(int).to_numpy()
    drop_cols = [c for c in df.columns if c.startswith("udps.")] + ["Label"]
    X_df = df.drop(columns=drop_cols, errors="ignore")
    X_df = X_df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0)
    return X_df.to_numpy(dtype=np.float32), y, list(X_df.columns)


def build_models(selected):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    out = {}
    if "rf" in selected:
        out["RandomForest"] = RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=42,
        )
    if "mlp" in selected:
        out["MLP"] = MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=80, random_state=42,
            early_stopping=True, validation_fraction=0.1,
        )
    if "xgb" in selected:
        try:
            from xgboost import XGBClassifier
            out["XGBoost"] = XGBClassifier(
                n_estimators=300, max_depth=8, learning_rate=0.1,
                n_jobs=-1, random_state=42, eval_metric="mlogloss",
            )
        except ImportError:
            print("[skip] xgboost not installed -- pip install xgboost")
    return out


def main():
    p = argparse.ArgumentParser(description="Flow-level baselines for XMF-GNN dataset")
    p.add_argument("--models", default="rf,mlp,xgb",
                   help="comma-separated subset of {rf, mlp, xgb}")
    p.add_argument("--held-out", default=None,
                   help="class to hold out of training for LOAO (e.g. Mirai)")
    p.add_argument("--train-csv", type=Path, default=TRAIN_CSV)
    p.add_argument("--test-csv",  type=Path, default=TEST_CSV)
    p.add_argument("--out-dir",   type=Path, default=OUT_DIR)
    args = p.parse_args()

    if not args.train_csv.exists() or not args.test_csv.exists():
        print(f"[err] missing CSVs: {args.train_csv} or {args.test_csv}")
        print("      run the main pipeline through step_balance first.")
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected = {s.strip() for s in args.models.split(",") if s.strip()}
    models = build_models(selected)
    if not models:
        print("[err] no models selected (--models rf,mlp,xgb)")
        sys.exit(1)

    print(f"loading train: {args.train_csv}")
    X_train, y_train, feat_names = load_flow_features(args.train_csv)
    print(f"loading test : {args.test_csv}")
    X_test,  y_test,  _          = load_flow_features(args.test_csv)
    print(f"  X_train={X_train.shape}  X_test={X_test.shape}  features={len(feat_names)}")

    benign_id = DEFAULT_CIC_IOT2023_LABELS["Benign"]

    # LOAO: hold a class out of TRAINING; keep it in TEST as the unseen class.
    held_out_id = None
    if args.held_out:
        if args.held_out not in DEFAULT_CIC_IOT2023_LABELS:
            print(f"[err] held-out class {args.held_out!r} not in label dict")
            sys.exit(1)
        held_out_id = DEFAULT_CIC_IOT2023_LABELS[args.held_out]
        keep = y_train != held_out_id
        X_train, y_train = X_train[keep], y_train[keep]
        print(f"  LOAO: held out class {args.held_out!r} (id={held_out_id}) "
              f"-> train shrunk to {X_train.shape[0]} rows")

    rows = []
    for name, model in models.items():
        print("\n" + "=" * 70)
        print(f"  {name}")
        print("=" * 70)
        t0 = time.time()
        model.fit(X_train, y_train)
        fit_t = time.time() - t0

        t0 = time.time()
        preds = model.predict(X_test)
        try:
            probs = model.predict_proba(X_test)
        except AttributeError:
            n_cls = int(max(y_train.max(), y_test.max())) + 1
            probs = np.eye(n_cls, dtype=np.float32)[preds]
        pred_t = time.time() - t0

        print(f"  fit={fit_t:.1f}s  predict={pred_t:.1f}s")
        calculate_metrics(preds, y_test)
        zd = zero_day_metrics(probs, y_test, benign_label=benign_id, verbose=True)

        row = {"model": name, "fit_s": fit_t, "predict_s": pred_t,
               "accuracy": float((preds == y_test).mean()),
               "held_out": args.held_out or ""}
        row.update({k: v for k, v in zd.items() if isinstance(v, (int, float))})

        if held_out_id is not None:
            unseen_mask = y_test == held_out_id
            if unseen_mask.any():
                pred_unseen = preds[unseen_mask]
                prob_unseen = probs[unseen_mask]
                attack_score = 1.0 - prob_unseen[:, benign_id]
                dr_argmax = float((pred_unseen != benign_id).mean())
                dr_thresh = float((attack_score >= 0.5).mean())
                print(f"\nLOAO -- held-out class {args.held_out!r}: {unseen_mask.sum()} samples")
                print(f"  DR (argmax != benign)         : {dr_argmax:.4f}")
                print(f"  DR (attack_score >= 0.50)     : {dr_thresh:.4f}")
                print(f"  mean attack_score             : {attack_score.mean():.4f}")
                row["DR_unseen_argmax"] = dr_argmax
                row["DR_unseen_threshold"] = dr_thresh
                row["mean_attack_score_unseen"] = float(attack_score.mean())

        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = args.out_dir / (
        f"baselines_{args.held_out}.csv" if args.held_out else "baselines.csv"
    )
    df.to_csv(out_path, index=False)
    print(f"\nresults written to {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
