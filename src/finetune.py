"""Fine-tuning demo: fine-tune T5-small for summarization on a small custom
subset of CNN/DailyMail.

This is a *demo*, not a full fine-tune. Full fine-tuning on all 287k
CNN/DailyMail articles needs a GPU and many hours; on this CPU-only,
~6 GB-RAM VM it is deliberately NOT attempted. Here T5-small (60M params,
the lightest encoder-decoder used in this task) is fine-tuned on
``--n`` articles (default 200) from results/train_sample_500.csv for
``--epochs`` (default 1), with inputs truncated to 512 tokens and targets
to 128 tokens. The resulting checkpoint is saved to
outputs/finetuned-t5-small-demo/ and can be used with cli.py, e.g.:

    python cli.py --example 1 --model outputs/finetuned-t5-small-demo

Usage:
    python -m src.finetune --n 200 --epochs 1
    python -m src.finetune --n 100 --epochs 1 --max-steps 10   # smoke test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

torch.set_num_threads(2)

ROOT = Path(__file__).parent.parent
BASE_MODEL = "models/google-t5/t5-small"
MAX_INPUT = 512
MAX_TARGET = 128


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--csv", default="results/train_sample_500.csv")
    ap.add_argument("--out", default="outputs/finetuned-t5-small-demo")
    args = ap.parse_args()

    df = pd.read_csv(ROOT / args.csv).head(args.n)
    print(f"Fine-tuning {BASE_MODEL} on {len(df)} articles, {args.epochs} epoch(s)")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

    def tokenize(batch):
        inputs = tokenizer(
            ["summarize: " + t for t in batch["article"]],
            max_length=MAX_INPUT, truncation=True,
        )
        targets = tokenizer(
            text_target=batch["highlights"],
            max_length=MAX_TARGET, truncation=True,
        )
        inputs["labels"] = targets["input_ids"]
        return inputs

    from datasets import Dataset  # local import: only needed here

    ds = Dataset.from_pandas(df[["article", "highlights"]])
    ds = ds.map(tokenize, batched=True, remove_columns=["article", "highlights"])
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    training_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=5e-5,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=ds,
                      data_collator=collator)
    result = trainer.train()
    print(f"Final training loss: {result.training_loss:.4f}")
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Saved fine-tuned model to {args.out}")


if __name__ == "__main__":
    main()
