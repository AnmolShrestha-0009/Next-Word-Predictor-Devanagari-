"""
Nepali Next-Word Predictor — Streamlit demo

Loads a trained LSTM (with embeddings initialized from pretrained NepaliBERT)
and shows top-N next-word predictions for any Nepali text input.

Expects a folder (default: ./demo_artifacts) containing:
    next_word_lstm.pt   - trained model weights (state_dict)
    vocab.json          - {"idx2word": [...]}
    config.json         - {"context_size", "embed_dim", "hidden_dim", "num_layers", "vocab_size"}

Produce these by running the "Save Artifacts" cell at the end of the notebook.

Run with:  streamlit run app.py
"""

import json
import re
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st

# Change this line (Line 23):
MODEL_DIR = r"C:\Users\anmol\Downloads\DEEP-lstm\Artifacts"

PAD, UNK = "<pad>", "<unk>"


# ---------- Tokenizer (must match the one used to build the vocab) ----------
def word_tokenize(sentence: str):
    return re.findall(r'[\u0900-\u097F]+|[a-zA-Z]+|\d+|[^\s\u0900-\u097Fa-zA-Z\d]', sentence)


# ---------- Model definition (must match training-time architecture) ----------
class NextWordLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim=256, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        return self.fc(output[:, -1, :])


@st.cache_resource
def load_artifacts(model_dir: str):
    with open(os.path.join("Artifacts", "config.json")) as f:
        config = json.load(f)
    with open(os.path.join("Artifacts", "vocab.json"), encoding="utf-8") as f:
        idx2word = json.load(f)["idx2word"]
    word2idx = {w: i for i, w in enumerate(idx2word)}

    model = NextWordLSTM(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
    )
    state_dict = torch.load(
        os.path.join(model_dir, "next_word_lstm.pt"), map_location="cpu"
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, idx2word, word2idx, config


def predict_next_word(text, model, idx2word, word2idx, context_size, top_n=5):
    tokens = word_tokenize(text)[-context_size:]
    ids = [word2idx.get(t, word2idx[UNK]) for t in tokens]
    ids = [word2idx[PAD]] * (context_size - len(ids)) + ids
    x = torch.tensor([ids], dtype=torch.long)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=-1).squeeze(0)

    top_probs, top_idx = torch.topk(probs, top_n)
    return tokens, [(idx2word[i.item()], p.item()) for p, i in zip(top_probs, top_idx)]


# ---------------------------- UI ----------------------------
st.set_page_config(page_title="Nepali Next-Word Predictor", page_icon="✍️", layout="centered")

st.title("✍️ Nepali Next-Word Predictor")

with st.expander("About this model", expanded=True):
    st.markdown(
        """
This is a lightweight next-word prediction model for Nepali text.

- **Corpus:** ~51k Nepali news articles (title + summary + body), cleaned and
  tokenized at the word level using a Devanagari-aware regex tokenizer.
- **Embeddings:** each vocabulary word's embedding was initialized from
  **pretrained NepaliBERT** (`Rajan/NepaliBERT`) by mean-pooling its WordPiece
  sub-token embeddings, then fine-tuned during training.
- **Architecture:** `Embedding → LSTM → Linear` over the vocabulary. Given the
  last few words as context, it predicts the most likely next word.
- **Output:** top-N candidate next words ranked by softmax probability.

This is a demo of the architecture, not a production-quality writing assistant —
predictions are only as good as the corpus and training budget behind them.
        """
    )

if not os.path.isdir(MODEL_DIR):
    st.warning(
        f"No trained model found in `./{MODEL_DIR}/`. Run the notebook's "
        "\"Save Artifacts\" cell after training, then drop the `{MODEL_DIR}/` "
        "folder next to this app.py."
    )
    st.stop()

model, idx2word, word2idx, config = load_artifacts(MODEL_DIR)

st.caption(
    f"Vocab size: {config['vocab_size']:,} · Context window: {config['context_size']} words"
)

text = st.text_input("Enter Nepali text", placeholder="नेपाल सरकारले आज")
top_n = st.slider("Number of predictions", min_value=1, max_value=10, value=5)

if st.button("Predict next word", type="primary") and text.strip():
    tokens, predictions = predict_next_word(
        text, model, idx2word, word2idx, config["context_size"], top_n
    )
    st.write(f"**Context used:** {' '.join(tokens)}")

    for rank, (word, prob) in enumerate(predictions, 1):
        col1, col2 = st.columns([3, 7])
        with col1:
            st.write(f"**{rank}. {word}**")
        with col2:
            st.progress(min(prob, 1.0), text=f"{prob:.1%}")
elif text.strip() == "":
    st.info("Type some Nepali text above and hit Predict.")
