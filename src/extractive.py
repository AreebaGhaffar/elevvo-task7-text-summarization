"""Extractive summarization bonus: TextRank (sumy) and a Lead-3 baseline.

Extractive methods do not generate new text — they pick the most important
sentences from the article itself. Two methods are compared on the same
subset used for the abstractive models:

1. TextRank (sumy ``TextRankSummarizer``): builds a sentence-similarity
   graph and ranks sentences with PageRank; the top ``n_sentences`` are
   returned in document order.
2. Lead-3: the first 3 sentences of the article. A famously strong baseline
   on CNN/DailyMail, because news articles front-load the key facts.

Both are scored with the same ROUGE-1/2/L (stemmed) as the abstractive
models, so the numbers in results/comparison.md are directly comparable.

Usage:
    python -m src.extractive --csv results/test_sample_100.csv --n 50 --sentences 3
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import nltk
import pandas as pd
from rouge_score import rouge_scorer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer

from src.summarize import preprocess

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"


def ensure_punkt() -> None:
    """Make sure the NLTK punkt sentence tokenizer is available.

    On this VM ``nltk.download`` is blocked (its SSRF check rejects the
    proxy-resolved IP of raw.githubusercontent.com), so if the download
    fails we fetch the same zip files with curl, unzip them into
    ~/nltk_data, and make the directory private (newer NLTK refuses to
    read data from a group-writable directory).
    """
    import shutil
    import subprocess

    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
            continue
        except LookupError:
            pass
        try:
            nltk.download(resource, quiet=True)
            nltk.data.find(f"tokenizers/{resource}")
            continue
        except Exception:
            pass
        dest = Path.home() / "nltk_data" / "tokenizers"
        dest.mkdir(parents=True, exist_ok=True)
        url = (f"https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/"
               f"packages/tokenizers/{resource}.zip")
        zip_path = dest / f"{resource}.zip"
        subprocess.run(["curl", "-sS", "-L", "-o", str(zip_path), url], check=True)
        shutil.unpack_archive(str(zip_path), str(dest))
        subprocess.run(["chmod", "-R", "go-w", str(Path.home() / "nltk_data")], check=True)
        nltk.data.find(f"tokenizers/{resource}")  # raises if still missing


def sentences(text: str) -> list[str]:
    return nltk.sent_tokenize(preprocess(text))


def textrank_summary(text: str, n_sentences: int = 3) -> str:
    parser = PlaintextParser.from_string(preprocess(text), Tokenizer("english"))
    summarizer = TextRankSummarizer()
    sents = summarizer(parser.document, n_sentences)
    return " ".join(str(s) for s in sents)


def lead_summary(text: str, n_sentences: int = 3) -> str:
    return " ".join(sentences(text)[:n_sentences])


def evaluate(df: pd.DataFrame, fn, name: str, n_sentences: int) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rows = []
    t0 = time.time()
    for _, row in df.iterrows():
        gen = fn(str(row["article"]), n_sentences)
        scores = scorer.score(str(row["highlights"]), gen)
        rows.append({
            "id": row["id"],
            "reference": row["highlights"],
            "generated": gen,
            "rouge1_f1": scores["rouge1"].fmeasure,
            "rouge2_f1": scores["rouge2"].fmeasure,
            "rougeL_f1": scores["rougeL"].fmeasure,
        })
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / f"evaluation_{name}.csv", index=False)
    summary = {
        "model": name,
        "family": "extractive",
        "n_examples": len(df),
        "n_sentences": n_sentences,
        "rouge1_f1": round(float(out["rouge1_f1"].mean()), 4),
        "rouge2_f1": round(float(out["rouge2_f1"].mean()), 4),
        "rougeL_f1": round(float(out["rougeL_f1"].mean()), 4),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    (RESULTS / f"evaluation_{name}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/test_sample_100.csv")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--sentences", type=int, default=3)
    args = ap.parse_args()
    ensure_punkt()
    df = pd.read_csv(ROOT / args.csv).iloc[args.offset : args.offset + args.n]
    print(f"Extractive evaluation on {len(df)} articles ({args.csv}, offset {args.offset})")
    evaluate(df, textrank_summary, "textrank", args.sentences)
    evaluate(df, lead_summary, "lead3", args.sentences)


if __name__ == "__main__":
    main()
