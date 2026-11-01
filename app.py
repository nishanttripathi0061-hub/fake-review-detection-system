from pathlib import Path
import os
import pickle

import numpy as np
import pandas as pd
import torch
import tensorflow as tf

from flask import Flask, render_template, request
from huggingface_hub import hf_hub_download, snapshot_download
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

LSTM_REPO_ID = "nishant-works/fake-review-lstm"
DISTILBERT_REPO_ID = "nishant-works/fake-review-distilbert"

PREDICTIONS_PATH = BASE_DIR / "data" / "test_predictions.csv"

app = Flask(__name__)

# Deployment will use CPU
device = torch.device("cpu")

# Reduce unnecessary CPU thread usage
torch.set_num_threads(2)


# ==================================================
# 2. DOWNLOAD AND LOAD LSTM MODEL
# ==================================================

print("\nDownloading LSTM model from Hugging Face...")

LSTM_MODEL_PATH = hf_hub_download(
    repo_id=LSTM_REPO_ID,
    filename="best_lstm_model.keras"
)

LSTM_TOKENIZER_PATH = hf_hub_download(
    repo_id=LSTM_REPO_ID,
    filename="lstm_tokenizer.pkl"
)

print("Loading LSTM model...")

lstm_model = tf.keras.models.load_model(
    LSTM_MODEL_PATH,
    compile=False
)

with open(LSTM_TOKENIZER_PATH, "rb") as file:
    lstm_tokenizer = pickle.load(file)

print("LSTM loaded successfully!")


# ==================================================
# 3. DOWNLOAD AND LOAD DISTILBERT MODEL
# ==================================================

print("\nDownloading DistilBERT model from Hugging Face...")

DISTILBERT_DIR = snapshot_download(
    repo_id=DISTILBERT_REPO_ID
)

print("Loading DistilBERT model...")

distilbert_tokenizer = AutoTokenizer.from_pretrained(
    DISTILBERT_DIR
)

distilbert_model = AutoModelForSequenceClassification.from_pretrained(
    DISTILBERT_DIR
).to(device)

distilbert_model.eval()

print("DistilBERT loaded successfully!")
print("Device:", device)


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
# 0 = Fake, 1 = Original
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

    # Probability of Original class (class 1)
    original_score = float(probabilities[1].item())

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "score": round(original_score, 4)
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
                        if agreement
                        else None
                    )
                }

            except Exception:

                app.logger.exception("Prediction error")

                error = "Prediction failed. Check server logs."

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
            "Evaluation file not found. "
            "Please ensure test_predictions.csv exists.",
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
                "Check test_predictions.csv.",
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
                    actual,
                    predicted,
                    zero_division=0
                ) * 100, 2
            ),

            "recall": round(
                recall_score(
                    actual,
                    predicted,
                    zero_division=0
                ) * 100, 2
            ),

            "f1": round(
                f1_score(
                    actual,
                    predicted,
                    zero_division=0
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
        y_true == 1,
        "Original",
        "Fake"
    )

    df["lstm_prediction"] = np.where(
        lstm_pred == 1,
        "Original",
        "Fake"
    )

    df["distilbert_prediction"] = np.where(
        bert_pred == 1,
        "Original",
        "Fake"
    )

    df["models_agree"] = agree

    error_count = int(
        (
            (lstm_pred != y_true)
            | (bert_pred != y_true)
        ).sum()
    )

    # Dashboard table loads the first 1,000 rows
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

        agreement_pct=round(
            float(agree.mean() * 100), 2
        ),

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

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )