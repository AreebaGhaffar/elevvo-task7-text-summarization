"""Abstractive summarization pipeline for Task 7.

Wraps the Hugging Face ``summarization`` pipeline around a pre-trained
encoder-decoder checkpoint (BART / DistilBART / T5 / Pegasus) and handles
the two things the task requires:

1. Preprocessing — strip the CNN/DailyMail artefacts (``(CNN)`` lead-ins,
   extra whitespace) and truncate the article to the model's input limit
   (1024 tokens for BART/Pegasus, 512 for T5-small), so long documents never
   overflow the encoder.
2. Generation — beam search with the model's own generation config where
   available (``facebook/bart-large-cnn`` ships one tuned for CNN/DM).

Runs on CPU only (``device=-1``). BART-large is the largest model used here
(~1.6 GB weights); on this 6 GB-RAM VM it must be run on its own, one
article at a time, with ``torch.set_num_threads(2)``.
"""
from __future__ import annotations

import re
import warnings

import torch
from transformers import pipeline

torch.set_num_threads(2)

# Model input limits (encoder tokens, *excluding* special tokens).
# BART/Pegasus add BOS+EOS (2 tokens) and their learned positional
# embeddings only cover 1024 positions, so a 1024-token input would crash
# with "index out of range" in the positional embedding — hence 1022.
# T5 adds only EOS (1 token) and uses relative positions, hence 511.
MAX_INPUT_TOKENS = {
    "bart": 1022,
    "pegasus": 1022,
    "t5": 511,
}

# Default generation lengths (in summary tokens). CNN/DM highlights average
# ~55 words, so 142 max / 56 min is the standard BART-CNN setting; T5-small
# gets a shorter cap because its 512-token input sees less of the article.
DEFAULT_GEN = {"max_length": 142, "min_length": 56}
T5_GEN = {"max_length": 128, "min_length": 30}


def model_family(model_path: str) -> str:
    name = model_path.lower()
    if "pegasus" in name:
        return "pegasus"
    if "t5" in name:
        return "t5"
    return "bart"  # bart-large-cnn and distilbart-cnn


def preprocess(text: str) -> str:
    """Clean an article: drop the ``(CNN)`` / ``(CNN) --`` lead-in, collapse
    whitespace, and strip stray newlines (highlights are joined with ' . ')."""
    text = re.sub(r"^\s*\(CNN\)\s*--?\s*", "", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_summarizer(model_path: str):
    """Load a summarization pipeline from a local checkpoint directory."""
    family = model_family(model_path)
    summarizer = pipeline(
        "summarization",
        model=model_path,
        tokenizer=model_path,
        device=-1,  # CPU
    )
    return summarizer, family


def truncate_to_limit(text: str, summarizer, family: str) -> tuple[str, int, int]:
    """Truncate ``text`` to the model's encoder limit. Returns
    (truncated_text, tokens_before, tokens_after)."""
    import logging

    tokenizer = summarizer.tokenizer
    limit = MAX_INPUT_TOKENS[family]
    # Tokenizing a >limit sequence logs a "sequence longer than maximum"
    # warning; silence just that logger for the counting call.
    logger = logging.getLogger("transformers.tokenization_utils_base")
    old_level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    finally:
        logger.setLevel(old_level)
    n_before = len(ids)
    if n_before > limit:
        ids = ids[:limit]
        text = tokenizer.decode(ids, skip_special_tokens=True)
    return text, n_before, len(ids)


def summarize(
    summarizer,
    family: str,
    text: str,
    max_length: int | None = None,
    min_length: int | None = None,
) -> dict:
    """Summarize one article. Returns the summary plus truncation info."""
    text = preprocess(text)
    text, n_before, n_after = truncate_to_limit(text, summarizer, family)
    gen = dict(T5_GEN if family == "t5" else DEFAULT_GEN)
    if max_length is not None:
        gen["max_length"] = max_length
    if min_length is not None:
        gen["min_length"] = min_length
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # T5 needs the "summarize: " task prefix; BART/Pegasus do not.
        inp = f"summarize: {text}" if family == "t5" else text
        # Use max_new_tokens / min_new_tokens only: passing max_length too
        # triggers a transformers warning (and some checkpoints set
        # max_new_tokens in their generation config, which would override
        # max_length anyway). The values are the same either way.
        out = summarizer(
            inp,
            max_new_tokens=gen["max_length"],
            min_new_tokens=gen["min_length"],
            do_sample=False,
        )
    return {
        "summary": out[0]["summary_text"].strip(),
        "input_tokens_before_truncation": n_before,
        "input_tokens_after_truncation": n_after,
        "truncated": n_before > n_after,
    }
