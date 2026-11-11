"""Download the CNN/DailyMail dataset for Task 7.

Recommended source (per the task): Kaggle
  https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail
  ("CNN-DailyMail News Text Summarization" — the CNN/DailyMail dataset,
  300k+ unique news articles, versions 2.0.0 / 3.0.0, columns id / article /
  highlights). The Kaggle download needs a Kaggle login, so this script first
  tries kagglehub (works if Kaggle credentials are configured) and otherwise
  falls back to the identical official files on the Hugging Face Hub
  (cnn_dailymail 3.0.0, parquet conversion), downloaded with curl — the same
  approach Task 6 used when the Hub downloader stalled.

What is saved (all in data/, gitignored):
  cnn_dailymail_test.csv          — full test split (11,490 articles) if the
                                    Kaggle/Parquet source provides it
  cnn_dailymail_train_sample.csv  — first shard of the train split, for the
                                    fine-tuning demo (small subset only)

Usage:  python download_data.py
"""
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

HF_BASE = "https://huggingface.co/api/datasets/abisee/cnn_dailymail/parquet/3.0.0"
# NOTE: the canonical CNN/DailyMail repo moved from `cnn_dailymail` to
# `abisee/cnn_dailymail` on the Hub; the parquet files are served by the
# datasets-server API at the URLs below (test = 1 shard, train = 3 shards;
# only the first train shard is downloaded, as a sample for the fine-tuning demo).


def curl(url: str, dest: Path) -> bool:
    print(f"Downloading {url} -> {dest} ...", flush=True)
    r = subprocess.run(["curl", "-sS", "-L", "-C", "-", "-o", str(dest), url])
    # Validate: a real parquet file starts with the magic bytes "PAR1".
    # (The Hub returns a short "Entry not found" body for wrong URLs.)
    ok = (
        r.returncode == 0
        and dest.exists()
        and dest.stat().st_size > 1000
        and dest.read_bytes()[:4] == b"PAR1"
    )
    if dest.exists():
        print(f"  {'OK' if ok else 'FAILED'}: {dest.stat().st_size:,} bytes")
    else:
        print("  FAILED")
    return ok


def parquet_to_csv(parquet_path: Path, csv_path: Path) -> bool:
    """Convert a cnn_dailymail parquet file to a CSV with article/highlights."""
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    keep = [c for c in ("id", "article", "highlights") if c in df.columns]
    df = df[keep]
    df.to_csv(csv_path, index=False)
    print(f"  -> {csv_path}: {len(df):,} rows, columns {list(df.columns)}")
    return True


def try_kaggle() -> bool:
    """Try the Kaggle dataset via kagglehub. Returns True on success."""
    try:
        import kagglehub
    except ImportError:
        print("kagglehub not installed; skipping Kaggle download.")
        return False
    try:
        print("Trying Kaggle: gowrishankarp/newspaper-text-summarization-cnn-dailymail ...")
        path = kagglehub.dataset_download(
            "gowrishankarp/newspaper-text-summarization-cnn-dailymail"
        )
    except Exception as e:  # noqa: BLE001 - auth / network failure -> fall back
        print(f"Kaggle download failed ({type(e).__name__}: {e}); falling back to Hugging Face.")
        return False
    import pandas as pd

    files = sorted(Path(path).glob("*.csv"))
    print(f"Kaggle files: {[f.name for f in files]}")
    if not files:
        return False
    # The dataset ships one CSV (11,490 rows) with id/article/highlights.
    df = pd.read_csv(files[0])
    cols = {c.lower(): c for c in df.columns}
    df = df.rename(columns={v: k for k, v in cols.items()})
    keep = [c for c in ("id", "article", "highlights") if c in df.columns]
    df[keep].to_csv(DATA_DIR / "cnn_dailymail_test.csv", index=False)
    print(f"  -> data/cnn_dailymail_test.csv: {len(df):,} rows")
    return True


def download_from_hf() -> bool:
    """Download test + first train shard parquet from the HF Hub, convert to CSV."""
    ok = True
    for split, fname in (("test", "test/0.parquet"), ("train", "train/0.parquet")):
        pq = DATA_DIR / f"cnn_dailymail_{split}.parquet"
        if not pq.exists() or pq.stat().st_size == 0:
            if not curl(f"{HF_BASE}/{fname}", pq):
                ok = False
                continue
        csv_name = (
            "cnn_dailymail_test.csv" if split == "test"
            else "cnn_dailymail_train_sample.csv"
        )
        parquet_to_csv(pq, DATA_DIR / csv_name)
    return ok


if __name__ == "__main__":
    done = try_kaggle()
    if not done:
        done = download_from_hf()
    print("Done." if done else "Download FAILED for all sources.")
    print("Files in data/:", sorted(p.name for p in DATA_DIR.iterdir()))
    sys.exit(0 if done else 1)
