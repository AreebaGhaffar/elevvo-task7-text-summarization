"""Evaluate a summarization model on CNN/DailyMail with ROUGE.

Loads a local checkpoint, generates summaries for the first ``--n`` rows of
a CSV (default: results/test_sample_100.csv), and scores them against the
reference highlights with ROUGE-1, ROUGE-2 and ROUGE-L (F1, plus precision
and recall). Results go to results/evaluation_<model>.json and
results/evaluation_<model>.csv.

ROUGE is computed with ``rouge_score`` (the standard implementation used by
the ``evaluate`` library's rouge metric), with stemming enabled.

Usage:
    python -m src.evaluate --model models/facebook/bart-large-cnn --n 50
    python -m src.evaluate --model models/google-t5/t5-small --n 100 --offset 0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from rouge_score import rouge_scorer

from src.summarize import load_summarizer, model_family, summarize

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"


def safe_name(model_path: str) -> str:
    return model_path.replace("models/", "").replace("/", "_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local checkpoint dir, e.g. models/facebook/bart-large-cnn")
    ap.add_argument("--csv", default="results/test_sample_100.csv")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--min-length", type=int, default=None)
    args = ap.parse_args()

    df = pd.read_csv(ROOT / args.csv).iloc[args.offset : args.offset + args.n]
    print(f"Evaluating {args.model} on {len(df)} articles ({args.csv}, offset {args.offset})")
    summarizer, family = load_summarizer(args.model)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rows = []
    t0 = time.time()
    for i, (_, row) in enumerate(df.iterrows()):
        res = summarize(summarizer, family, row["article"], args.max_length, args.min_length)
        scores = scorer.score(str(row["highlights"]), res["summary"])
        rows.append({
            "id": row["id"],
            "reference": row["highlights"],
            "generated": res["summary"],
            "rouge1_f1": scores["rouge1"].fmeasure,
            "rouge1_precision": scores["rouge1"].precision,
            "rouge1_recall": scores["rouge1"].recall,
            "rouge2_f1": scores["rouge2"].fmeasure,
            "rouge2_precision": scores["rouge2"].precision,
            "rouge2_recall": scores["rouge2"].recall,
            "rougeL_f1": scores["rougeL"].fmeasure,
            "rougeL_precision": scores["rougeL"].precision,
            "rougeL_recall": scores["rougeL"].recall,
            "truncated": res["truncated"],
        })
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(df)} done, {time.time() - t0:.0f}s elapsed", flush=True)
    elapsed = time.time() - t0

    out = pd.DataFrame(rows)
    name = safe_name(args.model)
    if args.offset:
        name += f"_offset{args.offset}"
    out.to_csv(RESULTS / f"evaluation_{name}.csv", index=False)
    summary = {
        "model": args.model,
        "family": family,
        "n_examples": len(df),
        "csv": args.csv,
        "offset": args.offset,
        "rouge1_f1": round(float(out["rouge1_f1"].mean()), 4),
        "rouge2_f1": round(float(out["rouge2_f1"].mean()), 4),
        "rougeL_f1": round(float(out["rougeL_f1"].mean()), 4),
        "rouge1_precision": round(float(out["rouge1_precision"].mean()), 4),
        "rouge1_recall": round(float(out["rouge1_recall"].mean()), 4),
        "rouge2_precision": round(float(out["rouge2_precision"].mean()), 4),
        "rouge2_recall": round(float(out["rouge2_recall"].mean()), 4),
        "rougeL_precision": round(float(out["rougeL_precision"].mean()), 4),
        "rougeL_recall": round(float(out["rougeL_recall"].mean()), 4),
        "elapsed_seconds": round(elapsed, 1),
        "seconds_per_article": round(elapsed / max(len(df), 1), 2),
        "n_truncated": int(out["truncated"].sum()),
    }
    (RESULTS / f"evaluation_{name}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
