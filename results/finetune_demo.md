# Fine-tuning demo — T5-small on a small CNN/DailyMail subset

## What was done

T5-small (60M params, the lightest encoder-decoder in this task) was
fine-tuned with the Hugging Face `Trainer` on a small custom subset of the
training split (`results/train_sample_500.csv`): inputs truncated to 512
tokens, targets to 128 tokens, batch size 2 with gradient accumulation 4,
learning rate 5e-5, 1 epoch. Command:

    python -m src.finetune --n 200 --epochs 1 --out outputs/finetuned-t5-small-demo

## Results

| Run | n | Steps | Train loss | Time |
|---|---|---|---|---|
| Smoke test | 100 | 10 (max-steps) | 2.7285 | 189 s |
| Demo | 200 | 25 (1 epoch) | 2.5744 | 425 s |

The fine-tuned checkpoint is saved to `outputs/finetuned-t5-small-demo/`
(gitignored — re-run the command above to reproduce it) and can be used
with the CLI, e.g. `python cli.py --example 1 --model outputs/finetuned-t5-small-demo`.

## What was NOT done

Full fine-tuning on all 287,113 CNN/DailyMail training articles was
deliberately NOT attempted: on this CPU-only, ~6 GB-RAM VM one epoch over
200 articles already takes ~7 minutes, so the full training set would
take days. The demo proves the training loop works end-to-end (loss
decreases: 2.73 -> 2.57) on a small subset.
