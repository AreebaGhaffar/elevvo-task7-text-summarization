"""Command-line interface for Task 7 text summarization.

Usage:
    # Summarize a built-in example article (article 1 or 2)
    python cli.py --example 1
    python cli.py --example 1 --model models/google-t5/t5-small

    # Summarize a file
    python cli.py --file examples/article1.txt
    python cli.py --file my_article.txt --model models/sshleifer/distilbart-cnn-12-6

    # Interactive: paste an article, get a summary
    python cli.py --interactive

    # Also show ROUGE against a reference summary file, if given
    python cli.py --example 1 --reference examples/article1_reference.txt

Default model: models/facebook/bart-large-cnn (best ROUGE in the comparison).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from rouge_score import rouge_scorer  # noqa: E402

from src.summarize import load_summarizer, summarize  # noqa: E402

DEFAULT_MODEL = "models/facebook/bart-large-cnn"


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize a long document (Task 7).")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--example", type=int, choices=[1, 2], default=None)
    ap.add_argument("--file", type=str, default=None)
    ap.add_argument("--reference", type=str, default=None)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--min-length", type=int, default=None)
    args = ap.parse_args()

    if args.interactive:
        print("Paste your article (end with an empty line, then Ctrl-D / Ctrl-Z):")
        text = sys.stdin.read()
    elif args.example:
        text = (ROOT / f"examples/article{args.example}.txt").read_text()
        if args.reference is None:
            args.reference = str(ROOT / f"examples/article{args.example}_reference.txt")
    elif args.file:
        text = Path(args.file).read_text()
    else:
        ap.error("give --example N, --file PATH, or --interactive")
    print(f"\nModel: {args.model}")
    print(f"Input: {len(text.split())} words")
    summarizer, family = load_summarizer(args.model)
    res = summarize(summarizer, family, text, args.max_length, args.min_length)
    print(f"\nSummary ({len(res['summary'].split())} words):")
    print(res["summary"])
    if res["truncated"]:
        print(f"\n[note: input truncated from {res['input_tokens_before_truncation']} "
              f"to {res['input_tokens_after_truncation']} tokens — the model's limit]")
    if args.reference:
        ref = Path(args.reference).read_text()
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(ref, res["summary"])
        print("\nROUGE vs reference:")
        for k in ("rouge1", "rouge2", "rougeL"):
            print(f"  {k}: F1 {scores[k].fmeasure:.4f} "
                  f"(P {scores[k].precision:.4f}, R {scores[k].recall:.4f})")
        print("\nReference:")
        print(ref)


if __name__ == "__main__":
    main()
