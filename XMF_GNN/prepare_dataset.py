"""Stream-extract CIC-IoT2023 PCAP.zip + run NFStream per class.

Why this exists
---------------
The CIC-IoT2023 zip is 244 GB and expands to ~548 GB. Most users don't have
that much free disk. This script:
  1. Reads PCAP.zip's TOC.
  2. For each of the 8 paper classes (Benign, WebBased, Spoofing, Recon,
     Mirai, DoS, DDoS, BruteForce), picks one PCAP file per CIC subfolder
     (or up to --pcaps-per-subfolder N).
  3. Extracts that PCAP to a tmp dir.
  4. Runs NFStream on it -> CSV with the proper class prefix.
  5. Deletes the PCAP. Keeps the CSV.
  6. Moves to the next file.

Peak disk: ~244 GB (zip) + one PCAP file at a time + accumulating CSVs.

After this script finishes, the CSVs end up in
    <project_root>/data/Extracted_Flow_Features/<PaperClass>-<original>.csv
which is exactly what run_pipeline.py step_balance expects.

Then run:
    ./run_all.sh --skip-extract        # already extracted by this script
"""
from __future__ import annotations
import argparse
import os
import re
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

# --------------------------------------------------------------------------
# Map zip top-level folder -> paper class (must match DEFAULT_CIC_IOT2023_LABELS)
# --------------------------------------------------------------------------
FOLDER_TO_CLASS = {
    # Benign
    "Benign_Final":            "Benign",
    # WebBased (6 sub-attacks)
    "Backdoor_Malware":        "WebBased",
    "BrowserHijacking":        "WebBased",
    "CommandInjection":        "WebBased",
    "SqlInjection":            "WebBased",
    "Uploading_Attack":        "WebBased",
    "XSS":                     "WebBased",
    # Spoofing (2 sub-attacks)
    "DNS_Spoofing":            "Spoofing",
    "MITM-ArpSpoofing":        "Spoofing",
    # Recon (5 sub-attacks)
    "Recon-HostDiscovery":     "Recon",
    "Recon-OSScan":            "Recon",
    "Recon-PingSweep":         "Recon",
    "Recon-PortScan":          "Recon",
    "VulnerabilityScan":       "Recon",
    # Mirai (3 sub-attacks)
    "Mirai-greeth_flood":      "Mirai",
    "Mirai-greip_flood":       "Mirai",
    "Mirai-udpplain":          "Mirai",
    # DoS (4 sub-attacks) -- use code form "Dos" (label_dict spelling)
    "DoS-HTTP_Flood":          "Dos",
    "DoS-SYN_Flood":           "Dos",
    "DoS-TCP_Flood":           "Dos",
    "DoS-UDP_Flood":           "Dos",
    # DDoS (12 sub-attacks) -- use code form "DDos" (label_dict spelling)
    "DDoS-ACK_Fragmentation":  "DDos",
    "DDoS-HTTP_Flood":         "DDos",
    "DDoS-ICMP_Flood":         "DDos",
    "DDoS-ICMP_Fragmentation": "DDos",
    "DDoS-PSHACK_Flood":       "DDos",
    "DDoS-RSTFINFlood":        "DDos",
    "DDoS-SlowLoris":          "DDos",
    "DDoS-SYN_Flood":          "DDos",
    "DDoS-SynonymousIP_Flood": "DDos",
    "DDoS-TCP_Flood":          "DDos",
    "DDoS-UDP_Flood":          "DDos",
    "DDoS-UDP_Fragmentation":  "DDos",
    # BruteForce
    "DictionaryBruteForce":    "BruteForce",
}

PROJECT_ROOT = Path(__file__).resolve().parent
EXTRACTOR    = PROJECT_ROOT / "Utility" / "Feature_extractor_flow_packet_combined.py"


def _natural_sort_key(name: str):
    """Sort BenignTraffic, BenignTraffic1, ..., BenignTraffic10 correctly."""
    parts = re.split(r"(\d+)", name)
    return [int(p) if p.isdigit() else p for p in parts]


def stream_process(zip_path: Path, out_csv_dir: Path, tmp_dir: Path,
                   pcaps_per_subfolder: int = 1,
                   dry_run: bool = False) -> None:
    out_csv_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"[prepare_dataset] zip      = {zip_path}")
    print(f"[prepare_dataset] tmp      = {tmp_dir}")
    print(f"[prepare_dataset] out CSV  = {out_csv_dir}")
    print(f"[prepare_dataset] policy   = {pcaps_per_subfolder} pcap(s) per CIC subfolder")
    print(f"[prepare_dataset] dry-run  = {dry_run}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Group PCAP entries by zip top-level folder.
        by_folder: dict[str, list[zipfile.ZipInfo]] = defaultdict(list)
        for info in zf.infolist():
            if not info.filename.endswith(".pcap"):
                continue
            top = info.filename.split("/")[0]
            if top in FOLDER_TO_CLASS:
                by_folder[top].append(info)

        # Sort each folder's PCAPs by filename then take the first N (usually
        # smallest because PCAP1 < PCAP2 < ... < PCAP10).
        # pcaps_per_subfolder <= 0 means "take all files" (paper-faithful).
        for folder, infos in by_folder.items():
            infos.sort(key=lambda i: _natural_sort_key(i.filename))
            if pcaps_per_subfolder > 0:
                by_folder[folder] = infos[:pcaps_per_subfolder]

        # Print plan
        total_bytes = 0
        for folder in sorted(by_folder):
            for info in by_folder[folder]:
                total_bytes += info.file_size
                print(f"  [{FOLDER_TO_CLASS[folder]:10s}] {info.filename} "
                      f"({info.file_size / 1e9:.2f} GB)")
        print(f"[prepare_dataset] total extract: {total_bytes / 1e9:.1f} GB "
              f"(across {sum(len(v) for v in by_folder.values())} pcap files)")

        if dry_run:
            print("[prepare_dataset] dry-run: not extracting.")
            return

        # Stream-process: extract one PCAP, run NFStream, delete PCAP.
        t_start = time.time()
        for folder in sorted(by_folder):
            cls = FOLDER_TO_CLASS[folder]
            for info in by_folder[folder]:
                base = Path(info.filename).stem  # eg "DDoS-ACK_Fragmentation"
                csv_name = f"{cls}-{base}.csv"
                csv_path = out_csv_dir / csv_name
                if csv_path.exists() and csv_path.stat().st_size > 0:
                    print(f"  [skip] {csv_name} already exists")
                    continue

                # Extract to tmp
                pcap_path = tmp_dir / Path(info.filename).name
                print(f"[extract] {info.filename} -> {pcap_path} "
                      f"({info.file_size / 1e9:.2f} GB)")
                t0 = time.time()
                with zf.open(info) as src, open(pcap_path, "wb") as dst:
                    while chunk := src.read(64 * 1024 * 1024):  # 64MB chunks
                        dst.write(chunk)
                print(f"           extracted in {time.time() - t0:.1f}s")

                # Run NFStream extractor
                # The extractor takes (pcap, out_dir_with_trailing_slash) and
                # writes <out_dir>/<pcap_basename>.csv. We then rename to add
                # the class prefix.
                tmp_csv_dir = tmp_dir / "csv"
                tmp_csv_dir.mkdir(exist_ok=True)
                print(f"[nfstream] processing {pcap_path.name}")
                t0 = time.time()
                try:
                    subprocess.check_call([
                        sys.executable, str(EXTRACTOR),
                        str(pcap_path), str(tmp_csv_dir) + os.sep,
                    ])
                except subprocess.CalledProcessError as e:
                    print(f"[ERROR] NFStream failed on {pcap_path}: {e}",
                          file=sys.stderr)
                    pcap_path.unlink(missing_ok=True)
                    continue
                print(f"           nfstream done in {time.time() - t0:.1f}s")

                # Move CSV to out dir with class-prefixed name
                produced = tmp_csv_dir / f"{pcap_path.stem}.csv"
                if produced.exists():
                    produced.rename(csv_path)
                    print(f"           csv: {csv_path}")
                else:
                    print(f"[WARN] no CSV produced for {pcap_path.name}")

                # Free disk
                pcap_path.unlink(missing_ok=True)

        # Cleanup tmp csv dir
        tmp_csv_dir = tmp_dir / "csv"
        try:
            tmp_csv_dir.rmdir()
        except OSError:
            pass

        elapsed = time.time() - t_start
        print(f"[prepare_dataset] FINISHED in {elapsed / 60:.1f} min")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zip", type=Path,
                   default=Path.home() / "Tutay/Tutay_Sec/CIC_IoT_2023_PCAP/PCAP.zip")
    p.add_argument("--out", type=Path,
                   default=PROJECT_ROOT / "data/Extracted_Flow_Features")
    p.add_argument("--tmp", type=Path,
                   default=Path.home() / "Tutay/Tutay_Sec/CIC_IoT_2023_PCAP/_tmp_extract")
    p.add_argument("--pcaps-per-subfolder", type=int, default=1,
                   help="how many PCAP files to take from each CIC subfolder "
                        "(default 1 = smallest only). Use 0 or -1 to take ALL "
                        "files (paper-faithful, slow). Increase for more "
                        "diversity per class.")
    p.add_argument("--dry-run", action="store_true",
                   help="just print the extraction plan and exit")
    args = p.parse_args()

    if not args.zip.exists():
        print(f"ERROR: zip not found at {args.zip}", file=sys.stderr)
        sys.exit(1)

    stream_process(args.zip, args.out, args.tmp,
                   pcaps_per_subfolder=args.pcaps_per_subfolder,
                   dry_run=args.dry_run)


if __name__ == "__main__":
    main()
