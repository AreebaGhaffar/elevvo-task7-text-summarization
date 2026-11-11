"""Download the summarization model checkpoints with curl.

Why curl, not the Hugging Face Hub downloader? On this VM the Hub
downloader stalls (Task 6 saw a 265 MB file come back as 0 bytes); plain
curl downloads the same files from the same Hub URLs at full speed.
Files land in models/<model-id>/ and can be used directly, e.g.:

    python -m src.evaluate --model models/facebook/bart-large-cnn --n 50

File lists were taken from the Hub API (https://huggingface.co/api/models/<id>)
on 2026-11-11. Where a model ships model.safetensors that is preferred over
pytorch_model.bin (same weights, half the download); distilbart and pegasus
only ship pytorch_model.bin.

Usage:  python download_models.py [model-id ...]
With no arguments, downloads all models used in the comparison.
"""
import subprocess
import sys
from pathlib import Path

BASE = "https://huggingface.co/{model}/resolve/main/{file}"
MODELS = {
    "facebook/bart-large-cnn": [
        "config.json", "generation_config.json", "merges.txt",
        "model.safetensors", "tokenizer.json", "vocab.json",
    ],
    "sshleifer/distilbart-cnn-12-6": [
        "config.json", "merges.txt", "pytorch_model.bin",
        "tokenizer_config.json", "vocab.json",
    ],
    "google-t5/t5-small": [
        "config.json", "generation_config.json", "model.safetensors",
        "spiece.model", "tokenizer.json", "tokenizer_config.json",
    ],
    "google/pegasus-cnn_dailymail": [
        "config.json", "generation_config.json", "pytorch_model.bin",
        "special_tokens_map.json", "spiece.model", "tokenizer_config.json",
    ],
}


def download(model: str):
    dest = Path("models") / model
    dest.mkdir(parents=True, exist_ok=True)
    for fname in MODELS[model]:
        url = BASE.format(model=model, file=fname)
        target = dest / fname
        # Download with resume + size verification. The egress proxy sometimes
        # drops the connection near the end of a large file while curl still
        # exits 0, leaving a truncated file — so verify the size against the
        # server's Content-Length and retry (resuming) until they match.
        for attempt in range(1, 6):
            print(f"{model}/{fname} (attempt {attempt}) ...", flush=True)
            r = subprocess.run(["curl", "-sS", "-L", "-C", "-", "-o", str(target), url])
            if r.returncode != 0:
                print(f"  curl failed (exit {r.returncode})")
                continue
            expected = expected_size(url)
            got = target.stat().st_size if target.exists() else 0
            if expected is None or got == expected:
                print(f"  OK: {got:,} bytes")
                break
            print(f"  incomplete: got {got:,} of {expected:,} bytes — resuming")
        else:
            print(f"  FAILED after 5 attempts: {fname}")
    files = sorted(p.name for p in dest.iterdir())
    print(f"  -> {dest}: {files}")
    # Sanity check: weights file must be non-trivial in size.
    weights = [f for f in files if f in ("model.safetensors", "pytorch_model.bin")]
    for w in weights:
        size = (dest / w).stat().st_size
        print(f"     {w}: {size:,} bytes {'OK' if size > 10_000_000 else 'SUSPICIOUSLY SMALL'}")


def expected_size(url: str) -> int | None:
    """Content-Length of a URL (following redirects), or None if unknown."""
    try:
        r = subprocess.run(
            ["curl", "-sIL", url], capture_output=True, text=True, timeout=60
        )
        lengths = [
            int(line.split(":", 1)[1].strip())
            for line in r.stdout.splitlines()
            if line.lower().startswith("content-length:")
        ]
        return lengths[-1] if lengths else None
    except Exception:
        return None


if __name__ == "__main__":
    for m in sys.argv[1:] or list(MODELS):
        download(m)
