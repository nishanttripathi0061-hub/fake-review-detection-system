
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "test.csv"
LSTM_PATH = BASE_DIR / "models" / "best_lstm_model.keras"
TOKENIZER_PATH = BASE_DIR / "models" / "lstm_tokenizer.pkl"
BERT_PATH = BASE_DIR / "models" / "distilbert-fake-review-model"

# Label mapping: 0 = Fake (CG), 1 = Original (OR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading test dataset...")
df = pd.read_csv(DATA_PATH)

print("Test dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# Use the same cleaned text used during training, if available
text_column = "clean_text" if "clean_text" in df.columns else "text_"

if text_column not in df.columns:
    raise ValueError("Neither clean_text nor text_ column exists.")

if "label_id" not in df.columns:
    raise ValueError("label_id column is missing from test.csv.")

df = df.dropna(subset=[text_column, "label_id"]).copy()
df[text_column] = df[text_column].astype(str)

y_true = df["label_id"].astype(int).to_numpy()
reviews = df[text_column].tolist()

if not set(np.unique(y_true)).issubset({0, 1}):
    raise ValueError("Expected label_id mapping: 0=Fake, 1=Original.")

print("Reviews to evaluate:", len(reviews))
print("Loading LSTM...")

lstm_model = tf.keras.models.load_model(
    str(LSTM_PATH), compile=False
)

with open(TOKENIZER_PATH, "rb") as f:
    lstm_tokenizer = pickle.load(f)

print("Loading DistilBERT...")

bert_tokenizer = AutoTokenizer.from_pretrained(
    str(BERT_PATH), local_files_only=True
)

bert_model = AutoModelForSequenceClassification.from_pretrained(
    str(BERT_PATH), local_files_only=True
).to(device)

bert_model.eval()

# LSTM predictions
print("Predicting with LSTM...")

sequences = lstm_tokenizer.texts_to_sequences(reviews)
padded = tf.keras.utils.pad_sequences(
    sequences,
    maxlen=250,
    padding="post",
    truncating="post"
)

lstm_scores = lstm_model.predict(
    padded, batch_size=128, verbose=1
).reshape(-1)

lstm_pred = (lstm_scores >= 0.5).astype(int)

# DistilBERT predictions
print("Predicting with DistilBERT...")

bert_predictions = []
batch_size = 16

with torch.no_grad():
    for start in range(0, len(reviews), batch_size):
        batch = reviews[start:start + batch_size]

        inputs = bert_tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        logits = bert_model(**inputs).logits
        predictions = torch.argmax(logits, dim=1)

        bert_predictions.extend(predictions.cpu().tolist())

bert_pred = np.array(bert_predictions)

# Metrics
def show_metrics(name, y_true, y_pred):
    print(f"\n{'=' * 45}")
    print(name)
    print("=" * 45)

    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1-score : {f1_score(y_true, y_pred, zero_division=0):.4f}")

    print("\nConfusion Matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Fake (CG)", "Original (OR)"],
        zero_division=0
    ))

show_metrics("LSTM", y_true, lstm_pred)
show_metrics("DistilBERT", y_true, bert_pred)

# Agreement analysis
agree = lstm_pred == bert_pred
agreement_count = int(agree.sum())
disagreement_count = int((~agree).sum())

print("\n" + "=" * 45)
print("MODEL AGREEMENT")
print("=" * 45)
print(f"Total evaluated: {len(y_true)}")
print(f"Both agree     : {agreement_count} ({agree.mean() * 100:.2f}%)")
print(f"Disagree       : {disagreement_count} ({(~agree).mean() * 100:.2f}%)")

# Save individual predictions for error analysis
df["actual_label_id"] = y_true
df["lstm_prediction_id"] = lstm_pred
df["distilbert_prediction_id"] = bert_pred
df["models_agree"] = agree

df["lstm_prediction"] = np.where(lstm_pred == 1, "Original", "Fake")
df["distilbert_prediction"] = np.where(bert_pred == 1, "Original", "Fake")
df["actual_label"] = np.where(y_true == 1, "Original", "Fake")

output_path = BASE_DIR / "data" / "test_predictions.csv"
df.to_csv(output_path, index=False)

print(f"\nPredictions saved to: {output_path}")
