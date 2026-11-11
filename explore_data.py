"""Explore the CNN/DailyMail data and save summary statistics.

Reads data/cnn_dailymail_test.csv (11,490 rows) and
data/cnn_dailymail_train_sample.csv (5,000 rows, first train shard trimmed).
Writes results/exploration_summary.md, results/test_sample_100.csv (first 100
test rows, used for ROUGE evaluation) and results/train_sample_500.csv
(first 500 train rows, used for the fine-tuning demo).

Usage:  python explore_data.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def word_count(text: str) -> int:
    return len(str(text).split())


def describe(df: pd.DataFrame, name: str) -> dict:
    art = df["article"].astype(str).map(word_count)
    hi = df["highlights"].astype(str).map(word_count)
    return {
        "name": name,
        "rows": len(df),
        "missing_article": int(df["article"].isna().sum()),
        "missing_highlights": int(df["highlights"].isna().sum()),
        "article_words_mean": round(float(art.mean()), 1),
        "article_words_median": int(art.median()),
        "article_words_min": int(art.min()),
        "article_words_max": int(art.max()),
        "highlights_words_mean": round(float(hi.mean()), 1),
        "highlights_words_median": int(hi.median()),
        "highlights_words_min": int(hi.min()),
        "highlights_words_max": int(hi.max()),
        "compression_ratio_mean": round(float((hi / art).mean()), 3),
    }


def main() -> None:
    test = pd.read_csv(ROOT / "data" / "cnn_dailymail_test.csv")
    train = pd.read_csv(ROOT / "data" / "cnn_dailymail_train_sample.csv")
    stats = [describe(test, "test split (full)"), describe(train, "train sample (first 5,000)")]

    # Evaluation / fine-tuning subsets (fixed slices, reproducible).
    test.head(100).to_csv(RESULTS / "test_sample_100.csv", index=False)
    train.head(500).to_csv(RESULTS / "train_sample_500.csv", index=False)

    lines = ["# CNN/DailyMail — exploration summary\n"]
    lines.append("Source: CNN/DailyMail 3.0.0 (identical to the Kaggle dataset "
                 "gowrishankarp/newspaper-text-summarization-cnn-dailymail), "
                 "downloaded from the Hugging Face Hub (`abisee/cnn_dailymail`) "
                 "because the Kaggle API is unreachable from this VM's proxy. "
                 "See download_data.py.\n")
    for s in stats:
        lines.append(f"## {s['name']}\n")
        lines.append(f"- Rows: {s['rows']:,}")
        lines.append(f"- Missing articles: {s['missing_article']}, missing highlights: {s['missing_highlights']}")
        lines.append(f"- Article length (words): mean {s['article_words_mean']}, "
                     f"median {s['article_words_median']}, min {s['article_words_min']}, "
                     f"max {s['article_words_max']}")
        lines.append(f"- Highlights length (words): mean {s['highlights_words_mean']}, "
                     f"median {s['highlights_words_median']}, min {s['highlights_words_min']}, "
                     f"max {s['highlights_words_max']}")
        lines.append(f"- Mean compression ratio (highlights / article): {s['compression_ratio_mean']}\n")
    lines.append("## Example (first test row)\n")
    row = test.iloc[0]
    lines.append("### Article (first 500 chars)\n")
    lines.append(str(row["article"])[:500] + "\n")
    lines.append("### Reference highlights\n")
    lines.append(str(row["highlights"]) + "\n")
    lines.append("## Subsets saved\n")
    lines.append("- `results/test_sample_100.csv` — first 100 test rows, used for ROUGE evaluation")
    lines.append("- `results/train_sample_500.csv` — first 500 train rows, used for the fine-tuning demo\n")
    (RESULTS / "exploration_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
