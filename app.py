"""Streamlit interface for Task 7 text summarization.

Run:  streamlit run app.py
Then paste a long document, pick a model, and get a summary (with an
optional ROUGE score if you also paste the reference summary).
"""
from pathlib import Path

import streamlit as st

from src.summarize import load_summarizer, summarize

ROOT = Path(__file__).parent
MODELS = {
    "BART-large-CNN (best ROUGE)": "models/facebook/bart-large-cnn",
    "DistilBART-CNN (faster, smaller)": "models/sshleifer/distilbart-cnn-12-6",
    "T5-small (lightest)": "models/google-t5/t5-small",
}

st.set_page_config(page_title="Task 7 — Text Summarization", layout="wide")
st.title("Text Summarization with Pre-trained Transformers")
st.write("Paste a long document (e.g. a news article) and get a concise "
         "abstractive summary. Models are pre-trained on CNN/DailyMail.")

model_label = st.selectbox("Model", list(MODELS))
col1, col2 = st.columns(2)
with col1:
    max_len = st.slider("Max summary length (tokens)", 30, 200, 142)
    min_len = st.slider("Min summary length (tokens)", 10, 100, 56)
with col2:
    use_example = st.button("Load example article")
    reference = st.text_area("Reference summary (optional — shows ROUGE)",
                             height=100)

if use_example:
    article = (ROOT / "examples/article1.txt").read_text()
else:
    article = st.text_area("Article", height=300,
                           placeholder="Paste a long article here...")

if st.button("Summarize") and article.strip():
    with st.spinner("Loading model and summarizing (CPU — may take a minute)..."):
        summarizer, family = load_summarizer(MODELS[model_label])
        res = summarize(summarizer, family, article, max_len, min_len)
    st.subheader("Summary")
    st.write(res["summary"])
    st.caption(f"{len(res['summary'].split())} words — input was "
               f"{len(article.split())} words"
               + (" (truncated to the model's input limit)" if res["truncated"] else ""))
    if reference.strip():
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, res["summary"])
        st.subheader("ROUGE vs reference")
        st.table({k: round(scores[k].fmeasure, 4) for k in ("rouge1", "rouge2", "rougeL")})
