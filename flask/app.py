"""
app.py
------
Flask application exposing a sentiment analysis API.

Endpoints:
    GET  /health         -> liveness check
    POST /predict         -> predict sentiment for a single sentence
    POST /predict_batch    -> predict sentiment for multiple sentences
                              (each with an optional timestamp), plus a chart
    POST /chart            -> generate a distribution chart from existing results
"""

import os
import io
import base64
import pickle
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, required for server-side rendering
import matplotlib.pyplot as plt
from flask_cors import CORS


from flask import Flask, request, jsonify

from preprocessing import encode_sentence, clean_single_text

# --------------------------------------------------------------------------
# Configuration (override via environment variables)
# --------------------------------------------------------------------------
MLFLOW_TRACKING_URI = "http://ec2-13-50-105-122.eu-north-1.compute.amazonaws.com:5000/"
MODEL_URI = "models:/BGRU/1"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WORD2IDX_PATH = os.path.join(BASE_DIR, "word2vec", "word2idx.json")
EMBEDDING_MATRIX_PATH = os.path.join(BASE_DIR, "word2vec", "embedding_matrix.npy")
MAX_LEN = 50

# Model output index -> human readable label
LABELS = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_ORDER = ["negative", "neutral", "positive"]
LABEL_COLORS = {"negative": "#e74c3c", "neutral": "#95a5a6", "positive": "#2ecc71"}

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = mlflow.pyfunc.load_model(MODEL_URI)

with open(WORD2IDX_PATH, "rb") as f:
    word2idx = json.load(f)

embedding_matrix = np.load(EMBEDDING_MATRIX_PATH)



def now_iso() -> str:
   
    return datetime.now(timezone.utc).isoformat()


def probabilities_to_label(logits):
    
    logits = np.array(logits).flatten()
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / np.sum(exp_logits)

    idx = int(np.argmax(probabilities))
    confidence = float(probabilities[idx])

    return LABELS.get(idx, "unknown"), confidence

def run_prediction(text: str) -> dict:
   
    cleaned = clean_single_text(text)
    encoded = encode_sentence(cleaned, MAX_LEN, word2idx)

    input_array = np.array([encoded])
    raw_output = model.predict(input_array)

    label, confidence = probabilities_to_label(raw_output)

    return {
        "text": text,
        "cleaned_text": cleaned,
        "sentiment": label,
        "confidence": round(confidence, 4),
    }


def validate_text_field(payload, key="text"):
    """Basic validation of an incoming JSON payload's text field."""
    if not payload or key not in payload:
        return None, f"Missing '{key}' field in request body."
    value = payload[key]
    if not isinstance(value, str) or not value.strip():
        return None, f"'{key}' must be a non-empty string."
    return value, None


def generate_chart(results: list) -> str:
    """
    Build a bar chart of sentiment distribution from a list of prediction
    dicts (each expected to have a 'sentiment' key). Returns a base64-encoded
    PNG string, ready to embed as `data:image/png;base64,<value>`.
    """
    df = pd.DataFrame(results)

    if "sentiment" not in df.columns or df.empty:
        counts = pd.Series(0, index=LABEL_ORDER)
    else:
        counts = df["sentiment"].value_counts().reindex(LABEL_ORDER, fill_value=0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values, color=[LABEL_COLORS[l] for l in counts.index])
    ax.set_title("Sentiment Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")

    for i, value in enumerate(counts.values):
        ax.text(i, value + max(counts.values) * 0.02 if counts.values.any() else 0.05,
                 str(value), ha="center")

    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("utf-8")



@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_uri": MODEL_URI}), 200


@app.route("/predict", methods=["POST"])
def predict_single():
    """
    Body: {"text": "I really love this product!"}
    """
    payload = request.get_json(silent=True)
    text, error = validate_text_field(payload)
    if error:
        return jsonify({"error": error}), 400

    try:
        result = run_prediction(text)
        result["timestamp"] = now_iso()
        return jsonify(result), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """
    Body:
    {
        "sentences": [
            {"text": "I love this!", "timestamp": "2026-07-29T10:00:00Z"},
            {"text": "This is bad."},
            "Plain string also works"
        ]
    }
    'timestamp' is optional per item; current UTC time is used if omitted.
    Returns predictions plus a base64 chart of the sentiment distribution.
    """
    payload = request.get_json(silent=True)
    if not payload or not isinstance(payload.get("sentences"), list):
        return jsonify({"error": "Request body must contain a 'sentences' list."}), 400

    items = payload["sentences"]
    if not items:
        return jsonify({"error": "'sentences' list cannot be empty."}), 400

    results = []
    for item in items:
        if isinstance(item, str):
            text, timestamp = item, None
        elif isinstance(item, dict):
            text, timestamp = item.get("text"), item.get("timestamp")
        else:
            continue

        if not text or not isinstance(text, str) or not text.strip():
            continue

        try:
            prediction = run_prediction(text)
        except Exception as exc:
            prediction = {"text": text, "error": str(exc)}

        prediction["timestamp"] = timestamp or now_iso()
        results.append(prediction)

    chart_base64 = generate_chart(results) if results else None

    return jsonify({
        "count": len(results),
        "results": results,
        "chart": chart_base64,
    }), 200


@app.route("/chart", methods=["POST"])
def chart_endpoint():
    """
    Body: {"results": [{"sentiment": "positive"}, {"sentiment": "negative"}, ...]}
    Standalone chart generation, useful if a client already has predictions
    and only wants the visualization.
    """
    payload = request.get_json(silent=True)
    if not payload or "results" not in payload:
        return jsonify({"error": "Request body must contain a 'results' list."}), 400

    chart_base64 = generate_chart(payload["results"])
    return jsonify({"chart": chart_base64}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
