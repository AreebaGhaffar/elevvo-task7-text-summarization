# CNN/DailyMail — exploration summary

Source: CNN/DailyMail 3.0.0 (identical to the Kaggle dataset gowrishankarp/newspaper-text-summarization-cnn-dailymail), downloaded from the Hugging Face Hub (`abisee/cnn_dailymail`) because the Kaggle API is unreachable from this VM's proxy. See download_data.py.

## test split (full)

- Rows: 11,490
- Missing articles: 0, missing highlights: 0
- Article length (words): mean 683.5, median 613, min 55, max 1954
- Highlights length (words): mean 55.0, median 51, min 9, max 564
- Mean compression ratio (highlights / article): 0.1

## train sample (first 5,000)

- Rows: 5,000
- Missing articles: 0, missing highlights: 0
- Article length (words): mean 615.2, median 570, min 18, max 1831
- Highlights length (words): mean 43.9, median 45, min 11, max 75
- Mean compression ratio (highlights / article): 0.093

## Example (first test row)

### Article (first 500 chars)

(CNN)The Palestinian Authority officially became the 123rd member of the International Criminal Court on Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian territories. The formal accession was marked with a ceremony at The Hague, in the Netherlands, where the court is based. The Palestinians signed the ICC's founding Rome Statute in January, when they also accepted its jurisdiction over alleged crimes committed "in the occupied Palestinian territory, includin

### Reference highlights

Membership gives the ICC jurisdiction over alleged crimes committed in Palestinian territories since last June .
Israel and the United States opposed the move, which could open the door to war crimes investigations against Israelis .

## Subsets saved

- `results/test_sample_100.csv` — first 100 test rows, used for ROUGE evaluation
- `results/train_sample_500.csv` — first 500 train rows, used for the fine-tuning demo
