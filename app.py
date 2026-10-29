
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch
import tensorflow as tf

from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

LSTM_MODEL_PATH = BASE_DIR / "models" / "best_lstm_model.keras"
LSTM_TOKENIZER_PATH = BASE_DIR / "models" / "lstm_tokenizer.pkl"
DISTILBERT_DIR = BASE_DIR / "models" / "distilbert-fake-review-model"
PREDICTIONS_PATH = BASE_DIR / "data" / "test_predictions.csv"

app = Flask(__name__)


# ==================================================
# 2. LOAD LSTM MODEL
# ==================================================

print("\nLoading LSTM model...")

if not LSTM_MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing: {LSTM_MODEL_PATH}")

if not LSTM_TOKENIZER_PATH.exists():
    raise FileNotFoundError(f"Missing: {LSTM_TOKENIZER_PATH}")

lstm_model = tf.keras.models.load_model(
    str(LSTM_MODEL_PATH),
    compile=False
)

with open(LSTM_TOKENIZER_PATH, "rb") as file:
    lstm_tokenizer = pickle.load(file)

print("LSTM loaded successfully!")


# ==================================================
# 3. LOAD DISTILBERT MODEL
# ==================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nLoading DistilBERT model...")

if not DISTILBERT_DIR.exists():
    raise FileNotFoundError(f"Missing: {DISTILBERT_DIR}")

distilbert_tokenizer = AutoTokenizer.from_pretrained(
    str(DISTILBERT_DIR),
    local_files_only=True
)

distilbert_model = AutoModelForSequenceClassification.from_pretrained(
    str(DISTILBERT_DIR),
    local_files_only=True
).to(device)

distilbert_model.eval()

print("DistilBERT loaded on:", device)


# ==================================================
# 4. LSTM PREDICTION
# 0 = Fake, 1 = Original
# ==================================================

def predict_lstm(review):

    sequence = lstm_tokenizer.texts_to_sequences([review])

    padded = tf.keras.utils.pad_sequences(
        sequence,
        maxlen=250,
        padding="post",
        truncating="post"
    )

    score = float(
        lstm_model.predict(padded, verbose=0)[0][0]
    )

    label = "Original" if score >= 0.5 else "Fake"
    confidence = max(score, 1 - score) * 100

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "score": round(score, 4)
    }


# ==================================================
# 5. DISTILBERT PREDICTION
# ==================================================

def predict_distilbert(review):

    inputs = distilbert_tokenizer(
        review,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = distilbert_model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )[0]

        predicted_class = int(
            torch.argmax(probabilities).item()
        )

    label = "Original" if predicted_class == 1 else "Fake"

    confidence = float(
        probabilities[predicted_class].item()
    ) * 100

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "score": round(confidence / 100, 4)
    }


# ==================================================
# 6. HOME PAGE: REVIEW ANALYZER
# ==================================================

@app.route("/", methods=["GET", "POST"])
def home():

    review = ""
    lstm_result = None
    distilbert_result = None
    comparison = None
    error = None

    if request.method == "POST":

        review = request.form.get("review", "").strip()

        if not review:
            error = "Please enter a review."

        else:
            try:
                lstm_result = predict_lstm(review)
                distilbert_result = predict_distilbert(review)

                agreement = (
                    lstm_result["label"]
                    == distilbert_result["label"]
                )

                comparison = {
                    "agreement": agreement,
                    "message": (
                        "Both models predicted the same label."
                        if agreement
                        else "Models disagree. Review carefully."
                    ),
                    "label": (
                        lstm_result["label"]
                        if agreement else None
                    )
                }

            except Exception:
                app.logger.exception("Prediction error")
                error = "Prediction failed. Check terminal output."

    return render_template(
        "index.html",
        review=review,
        lstm_result=lstm_result,
        distilbert_result=distilbert_result,
        comparison=comparison,
        error=error
    )


# ==================================================
# 7. EVALUATION DASHBOARD
# ==================================================

@app.route("/dashboard")
def dashboard():

    if not PREDICTIONS_PATH.exists():
        return (
            "Evaluation file not found. Run "
            "'python evaluate_models.py' first.",
            404
        )

    df = pd.read_csv(PREDICTIONS_PATH)

    required_columns = [
        "actual_label_id",
        "lstm_prediction_id",
        "distilbert_prediction_id"
    ]

    for column in required_columns:
        if column not in df.columns:
            return (
                f"Missing column: {column}. "
                "Run evaluate_models.py again.",
                400
            )

    y_true = df["actual_label_id"].astype(int).to_numpy()
    lstm_pred = df["lstm_prediction_id"].astype(int).to_numpy()
    bert_pred = df["distilbert_prediction_id"].astype(int).to_numpy()

    def calculate_metrics(actual, predicted):

        return {
            "accuracy": round(
                accuracy_score(actual, predicted) * 100, 2
            ),
            "precision": round(
                precision_score(
                    actual, predicted, zero_division=0
                ) * 100, 2
            ),
            "recall": round(
                recall_score(
                    actual, predicted, zero_division=0
                ) * 100, 2
            ),
            "f1": round(
                f1_score(
                    actual, predicted, zero_division=0
                ) * 100, 2
            ),
            "matrix": confusion_matrix(
                actual,
                predicted,
                labels=[0, 1]
            ).tolist()
        }

    agree = lstm_pred == bert_pred

    df["actual_label"] = np.where(
        y_true == 1, "Original", "Fake"
    )

    df["lstm_prediction"] = np.where(
        lstm_pred == 1, "Original", "Fake"
    )

    df["distilbert_prediction"] = np.where(
        bert_pred == 1, "Original", "Fake"
    )

    df["models_agree"] = agree

    error_count = int(
        ((lstm_pred != y_true) | (bert_pred != y_true)).sum()
    )

    # Dashboard table loads the first 1,000 rows.
    records = (
        df.head(1000)
        .fillna("")
        .to_dict(orient="records")
    )

    return render_template(
        "dashboard.html",
        total=len(df),
        agreement_count=int(agree.sum()),
        disagreement_count=int((~agree).sum()),
        agreement_pct=round(float(agree.mean() * 100), 2),
        error_count=error_count,
        lstm=calculate_metrics(y_true, lstm_pred),
        bert=calculate_metrics(y_true, bert_pred),
        records=records
    )


# ==================================================
# 8. START FLASK SERVER
# ==================================================

if __name__ == "__main__":

    print("\nFake Review Detection System")
    print("Review Analyzer: http://127.0.0.1:5000")
    print("Evaluation Dashboard: http://127.0.0.1:5000/dashboard")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
