# Model comparison

All models were evaluated on the same test subset — the first 50 test articles — except Pegasus, which was evaluated on the first 20 (28 s/article on CPU, so 50 would have taken ~25 min).

| Model | Type | n | ROUGE-1 | ROUGE-2 | ROUGE-L | s/article |
|---|---|---|---|---|---|---|
| models/facebook/bart-large-cnn | bart | 50 | 0.3492 | 0.1603 | 0.2603 | 20.9 |
| models/sshleifer/distilbart-cnn-12-6 | bart | 50 | 0.3440 | 0.1530 | 0.2557 | 13.9 |
| models/google-t5/t5-small | t5 | 50 | 0.3242 | 0.1291 | 0.2375 | 3.8 |
| models/google/pegasus-cnn_dailymail | pegasus | 20 | 0.3645 | 0.1544 | 0.2609 | 28.0 |
| textrank | extractive | 50 | 0.2335 | 0.0740 | 0.1623 | — |
| lead3 | extractive | 50 | 0.2978 | 0.1265 | 0.2170 | — |

## Reading the results

- All scores are ROUGE F1 (stemmed) against the reference highlights.
- Pegasus leads on ROUGE-1 (0.3645) and ROUGE-L (0.2609), but it was scored on only the first 20 articles, so its numbers are not strictly comparable to the n=50 rows; BART-large leads on ROUGE-2 (0.1603) among the n=50 models, and is the best overall choice when speed matters too (20.9 s/article vs Pegasus 28.0).
- DistilBART is close behind BART-large on every metric (ROUGE-1 0.3440 vs 0.3492) at ~1.5x the speed, and T5-small is the lightest model (ROUGE-1 0.3242) — but its 511-token input limit truncated 35 of 50 articles, so it sees less of each article.
- TextRank and Lead-3 are extractive: they copy sentences from the article instead of generating new text. Lead-3 is a famously strong baseline on news, because articles front-load the key facts.