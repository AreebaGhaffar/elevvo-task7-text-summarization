"""Build the model comparison table from the per-model evaluation JSONs.

Reads results/evaluation_*.json (abstractive models + TextRank + Lead-3),
writes results/comparison.md and results/comparison.csv. Re-run after every
evaluation to keep the table in sync.

Usage:  python -m src.compare_models
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"

ORDER = [
    "facebook/bart-large-cnn",
    "sshleifer/distilbart-cnn-12-6",
    "google-t5/t5-small",
    "google/pegasus-cnn_dailymail",
    "textrank",
    "lead3",
]


def main() -> None:
    rows = []
    for f in sorted(RESULTS.glob("evaluation_*.json")):
        d = json.loads(f.read_text())
        rows.append({
            "model": d["model"],
            "family": d.get("family", "?"),
            "n": d["n_examples"],
            "rouge1": d["rouge1_f1"],
            "rouge2": d["rouge2_f1"],
            "rougeL": d["rougeL_f1"],
            "seconds_per_article": d.get("seconds_per_article"),
            "elapsed_seconds": d.get("elapsed_seconds"),
        })
    df = pd.DataFrame(rows)
    df["order"] = df["model"].apply(lambda m: ORDER.index(m) if m in ORDER else 99)
    df = df.sort_values(["order", "model"]).drop(columns="order")
    df.to_csv(RESULTS / "comparison.csv", index=False)

    lines = ["# Model comparison (same test subset: first 50 test articles)\n"]
    lines.append("| Model | Type | n | ROUGE-1 | ROUGE-2 | ROUGE-L | s/article |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        s = f"{r['seconds_per_article']:.1f}" if pd.notna(r["seconds_per_article"]) else "—"
        lines.append(f"| {r['model']} | {r['family']} | {int(r['n'])} | "
                     f"{r['rouge1']:.4f} | {r['rouge2']:.4f} | {r['rougeL']:.4f} | {s} |")
    lines.append("")
    lines.append("## Reading the results")
    lines.append("")
    lines.append("- All scores are ROUGE F1 (stemmed) against the reference highlights.")
    lines.append("- BART-large-CNN is the standard CNN/DailyMail checkpoint and is "
                 "expected to lead; DistilBART trades a little ROUGE for ~2x speed; "
                 "T5-small is the lightest model and its 512-token input limit means "
                 "it sees less of each article.")
    lines.append("- TextRank and Lead-3 are extractive: they copy sentences from the "
                 "article instead of generating new text. Lead-3 is a famously strong "
                 "baseline on news, because articles front-load the key facts.")
    (RESULTS / "comparison.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
