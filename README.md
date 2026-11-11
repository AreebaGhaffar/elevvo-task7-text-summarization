# Task 7 — Text Summarization Using Pre-trained Models

Generate concise summaries from long news documents with pre-trained
encoder-decoder transformers (BART, T5, Pegasus), evaluated with ROUGE on
the CNN/DailyMail dataset. Built directly, without Claude, following the
same step-by-step approach as Task 6.

## Dataset

CNN/DailyMail 3.0.0 — 300k+ unique news articles from CNN and the Daily
Mail, each with human-written highlights as the reference summary.

- Recommended source (per the task): Kaggle,
  https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail
- Actually used: the identical official files from the Hugging Face Hub
  (`abisee/cnn_dailymail`, the canonical repo the old `cnn_dailymail`
  name now redirects to), because the Kaggle API (`api.kaggle.com`) is
  unreachable from this VM's egress proxy. See `download_data.py`.
- Test split: **11,490 articles**; train sample: first 5,000 rows of the
  first train shard (the full train split is 287,113 articles).
- Articles average **~684 words** (median 613, max 1,954); highlights
  average **~55 words** — a mean compression ratio of ~0.10. No missing
  values. Full stats in `results/exploration_summary.md`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python download_data.py      # ~300 MB (test + first train shard, parquet -> CSV)
python download_models.py    # ~5.3 GB (BART-large, DistilBART, T5-small, Pegasus)
```

`transformers` is pinned to `4.57.1`: v5 removed the `summarization`
pipeline. Model files are downloaded with `curl`, not the Hub downloader,
because on this VM the Hub downloader stalls and returns truncated files;
`download_models.py` verifies every file against the server's
Content-Length and resumes until it matches. Data, models and outputs are
gitignored — re-download them with the scripts above after cloning.

NLTK's `punkt` sentence tokenizer is needed for the extractive bonus.
`nltk.download()` is blocked on this VM (its SSRF check rejects the
proxy-resolved IP), so `src/extractive.py` falls back to downloading the
same zip files with `curl` into `~/nltk_data`.

## Usage

```bash
# Summarize a built-in example (BART-large by default)
python cli.py --example 1

# Summarize your own file, with another model
python cli.py --file my_article.txt --model models/google-t5/t5-small

# Interactive: paste an article
python cli.py --interactive

# Streamlit app
streamlit run app.py
```

Long texts are preprocessed (CNN lead-ins and extra whitespace stripped)
and **truncated to the model's input limit** — 1022 tokens for BART /
Pegasus, 511 for T5-small — so long documents never overflow the encoder.
(The limits leave room for the BOS/EOS special tokens: truncating BART
to exactly 1024 crashed with an `index out of range` in the positional
embedding.)

## Results (ROUGE F1, stemmed, first 50 test articles unless noted)

| Model | Type | n | ROUGE-1 | ROUGE-2 | ROUGE-L | s/article |
|---|---|---|---|---|---|---|
| BART-large-CNN | abstractive | 50 | 0.3492 | 0.1603 | 0.2603 | 20.9 |
| DistilBART-CNN-12-6 | abstractive | 50 | 0.3440 | 0.1530 | 0.2557 | 13.9 |
| T5-small | abstractive | 50 | 0.3242 | 0.1291 | 0.2375 | 3.8 |
| Pegasus-CNN/DailyMail | abstractive | 20 | 0.3645 | 0.1544 | 0.2609 | 28.0 |
| Lead-3 (first 3 sentences) | extractive | 50 | 0.2978 | 0.1265 | 0.2170 | — |
| TextRank | extractive | 50 | 0.2335 | 0.0740 | 0.1623 | — |

Full table and reading notes: `results/comparison.md`;
per-model details: `results/evaluation_*.json` / `.csv`;
combined: `results/all_evaluations.csv`.

**Reading the results:** Pegasus leads on ROUGE-1 and ROUGE-L, but on only
the first 20 articles, so it is not strictly comparable to the n=50 rows.
Among the n=50 models BART-large leads on every metric and is the best
overall choice; DistilBART is close behind at ~1.5x the speed, and
T5-small is the lightest and fastest (3.8 s/article) but its 511-token
limit truncated 35 of 50 articles. Lead-3 beats TextRank — news articles
front-load the key facts. All scores are in line with published
CNN/DailyMail results for these checkpoints (e.g. BART-large ~0.44
ROUGE-1 on the full test set with the standard 1024-token / beam-4
settings is higher; the gap here comes from the small 50-article slice
and CPU beam settings).

## Fine-tuning demo (bonus)

T5-small was fine-tuned on a small custom subset (200 train articles,
1 epoch, batch 2 + grad-accum 4, lr 5e-5):

```bash
python -m src.finetune --n 200 --epochs 1 --out outputs/finetuned-t5-small-demo
```

Train loss **2.5744** (smoke test on 100 articles / 10 steps: 2.7285).
The fine-tuned checkpoint works with the CLI
(`python cli.py --example 1 --model outputs/finetuned-t5-small-demo` —
ROUGE-1 0.4167 on example 1, vs the base T5-small's 0.3242 average).
Full fine-tuning on all 287k training articles was **deliberately NOT
attempted** on this CPU-only, ~6 GB-RAM machine: one epoch over 200
articles already takes ~7 minutes. Details in
`results/finetune_demo.md`.

## Project structure

```
download_data.py      download CNN/DailyMail (Kaggle first, HF parquet fallback)
download_models.py    download checkpoints with curl (+ size verification)
explore_data.py       dataset statistics -> results/exploration_summary.md
cli.py                command-line summarizer (example / file / interactive)
app.py                Streamlit interface (model picker, ROUGE vs reference)
src/
  summarize.py        pipeline wrapper: preprocessing, truncation, generation
  evaluate.py         ROUGE evaluation on a test subset
  compare_models.py   builds results/comparison.md/.csv from evaluation JSONs
  extractive.py       TextRank (sumy) + Lead-3 extractive baselines
  finetune.py         CPU-friendly T5-small fine-tuning demo
examples/             2 test articles + their reference highlights
results/              exploration summary, evaluation JSONs/CSVs, comparison,
                      fine-tune notes, CLI example outputs
```

## Limitations

- All evaluations use small subsets (n=50, Pegasus n=20) because
  generation on CPU takes 4–28 s per article; treat the numbers as
  same-subset comparisons, not full-test-set scores.
- Pegasus was evaluated on fewer articles for the same reason.
- Models, data and fine-tuned checkpoints are not in the repo
  (gitignored, ~6 GB); re-download with the scripts above.
- Summaries are extractive-leaning and can miss facts from the tail of
  very long articles, which are truncated to the input limit.

## Git history

9 commits on `master`, one per step (dataset → pipeline → T5 eval →
DistilBART eval → comparison → extractive → fine-tune demo → CLI/Streamlit
→ README). See `git log --oneline`.
