# Fake Review Detection System

A machine-learning-based web application that classifies product reviews as **Fake (Computer Generated)** or **Original** using two deep-learning models: **LSTM** and **DistilBERT**.

The project includes a live Streamlit application, a Flask-based local application, and an interactive model evaluation dashboard.

> **Caution:** Model predictions are estimates, not proof that a real-world review is fake or genuine.

## Live Demo

**Streamlit Application:**  
https://fake-review-detection-system-vwmy8h9a2sfhccdcfzxeva.streamlit.app/

## Table of Contents

- Overview
- Features
- Technology Stack
- Dataset and Labels
- Project Workflow
- Models and Architecture
- Model Evaluation Results
- Confusion Matrices
- Model Agreement
- Web Applications
- Project Structure
- Installation and Setup
- Limitations
- Future Improvements

## Overview

The system compares two text-classification approaches:

1. **LSTM (Long Short-Term Memory):** A recurrent neural network for sequential text processing.
2. **DistilBERT:** A transformer-based sequence-classification model.

Users can submit a product review and receive predictions from both models. The project also provides an evaluation dashboard for comparing model performance, confusion matrices, prediction agreement, and individual test examples.

## Features

- Fake and Original review classification.
- Dual-model prediction using LSTM and DistilBERT.
- Prediction confidence display.
- Model agreement and disagreement detection.
- Interactive evaluation dashboard.
- Accuracy, precision, recall, and F1-score comparison.
- Confusion matrix visualization.
- Searchable test prediction records.
- Streamlit-based deployed application.
- Flask-based local web application.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow / Keras | LSTM model |
| PyTorch | DistilBERT inference |
| Hugging Face Transformers | Transformer model and tokenizer |
| Pandas | Data handling |
| NumPy | Numerical operations |
| Scikit-learn | Evaluation metrics |
| Streamlit | Deployed web application |
| Flask | Local web application |
| HTML / CSS | Frontend structure and styling |
| JavaScript | Frontend interactions |
| Chart.js | Dashboard charts |

## Dataset and Labels

| Item | Details |
|---|---|
| Raw dataset | `data/fake reviews dataset.csv` |
| Training set | `data/train.csv` |
| Validation set | `data/validation.csv` |
| Test set | `data/test.csv` |
| Evaluation output | `data/test_predictions.csv` |
| Dataset size | Approximately 40,432 reviews |
| Test set size | 6,059 reviews |
| Fake label | CG (0) |
| Original label | OR (1) |
| Split | 70% / 15% / 15% |
| Strategy | Stratified split |

The evaluation script reportedly selects `clean_text` if available; otherwise, it uses `text_`.

## Project Workflow

1. Load the raw review dataset.
2. Analyze and preprocess the review text.
3. Split the dataset into training, validation, and test sets.
4. Train and evaluate the LSTM model.
5. Train/evaluate or load the saved DistilBERT model.
6. Run both models on the held-out test set.
7. Save predictions in `data/test_predictions.csv`.
8. Compare model performance.
9. Integrate model inference into the web applications.

## Models and Architecture

### 1. LSTM

- Framework: TensorFlow / Keras
- Saved model: `models/best_lstm_model.keras`
- Saved tokenizer: `models/lstm_tokenizer.pkl`
- Input sequence length: 250 tokens
- Padding and truncation: Post-padding and post-truncation
- Output: Sigmoid score
- Class mapping: 0 = Fake, 1 = Original
- Threshold: 0.5

### 2. DistilBERT

- Framework: PyTorch
- Library: Hugging Face Transformers
- Model directory: `models/distilbert-fake-review-model/`
- Maximum token length: 256
- Class mapping: 0 = Fake, 1 = Original
- Output: Softmax probabilities
- Device: CUDA when available, otherwise CPU

### Prediction Comparison

Both models analyze the submitted review.

- Matching predictions produce an agreement message.
- Different predictions produce a disagreement warning.

Agreement does not guarantee correctness.

## Model Evaluation Results

Reported results on the held-out test set of **6,059 reviews**:

| Metric | LSTM | DistilBERT |
|---|---:|---:|
| Accuracy | 93.46% | 97.95% |
| Precision | 93.33% | 99.06% |
| Recall | 93.63% | 96.83% |
| F1-score | 93.48% | 97.93% |
| Correct predictions | 5,663 | 5,935 |
| Incorrect predictions | 396 | 124 |

**Result:** DistilBERT outperformed LSTM on the reported aggregate metrics, producing 272 more correct predictions and 272 fewer incorrect predictions.

*Note: Precision, recall, and F1-score depend on the evaluation code's averaging and positive-label settings.*

## Confusion Matrices

Class order: Fake (0), Original (1). Rows represent actual labels and columns represent predicted labels.

### LSTM

| Actual / Predicted | Fake (0) | Original (1) |
|---|---:|---:|
| Fake (0) | 2,824 | 203 |
| Original (1) | 193 | 2,839 |

- Correct predictions: 5,663
- Incorrect predictions: 396

### DistilBERT

| Actual / Predicted | Fake (0) | Original (1) |
|---|---:|---:|
| Fake (0) | 2,999 | 28 |
| Original (1) | 96 | 2,936 |

- Correct predictions: 5,935
- Incorrect predictions: 124

## Model Agreement

| Measure | Result |
|---|---:|
| Both models agree | 5,647 reviews |
| Agreement rate | 93.20% |
| Models disagree | 412 reviews |
| Disagreement rate | 6.80% |

Agreement is calculated by comparing `lstm_prediction_id` with `distilbert_prediction_id`.

## Web Applications

### 1. Streamlit Application — Primary Deployment

**Live URL:**  
https://fake-review-detection-system-vwmy8h9a2sfhccdcfzxeva.streamlit.app/

The Streamlit application is the primary deployed interface for review predictions and model evaluation.

### 2. Flask Application — Local Version

The Flask application provides:

- Review analyzer at `/`
- Evaluation dashboard at `/dashboard`
- LSTM and DistilBERT predictions
- Prediction confidence
- Model agreement information
- Metrics and confusion matrix visualizations
- Searchable test prediction records

Local URLs:

- Analyzer: `http://127.0.0.1:5000/`
- Dashboard: `http://127.0.0.1:5000/dashboard`

## Project Structure

```text
fake review detection system/
│
├── app.py
├── evaluate_models.py
├── requirements.txt
│
├── streamlit_deploy/
│   ├── streamlit_app.py
│   └── requirements.txt
│
├── data/
│   ├── fake reviews dataset.csv
│   ├── train.csv
│   ├── validation.csv
│   ├── test.csv
│   └── test_predictions.csv
│
├── models/
│   ├── best_lstm_model.keras
│   ├── lstm_tokenizer.pkl
│   └── distilbert-fake-review-model/
│
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_lstm_model.ipynb
│   ├── 04_distilbert_model.ipynb
│   └── 05_model_comparison.ipynb
│
└── templates/
    ├── index.html
    └── dashboard.html
```

*This is the core project structure; update it if additional frontend or deployment files are added.*

## Installation and Setup

### Prerequisites

- Python
- Git
- VS Code (recommended)

### 1. Clone the repository

```bash
git clone https://github.com/nishanttripathi0061-hub/fake-review-detection-system.git

cd fake-review-detection-system
```

### 2. Run the Streamlit Application

Install the deployment dependencies:

```bash
python -m pip install -r streamlit_deploy/requirements.txt
```

Run the application:

```bash
streamlit run streamlit_deploy/streamlit_app.py
```

### 3. Run the Flask Application

Activate the Flask environment and install the root requirements if needed.

```powershell
& "D:\fake-review-env\Scripts\Activate.ps1"

python app.py
```

Open:

- `http://127.0.0.1:5000/`
- `http://127.0.0.1:5000/dashboard`

Keep the terminal open while the Flask server is running. Stop it using `Ctrl + C`.

### Evaluation Predictions

The Flask dashboard requires:

`data/test_predictions.csv`

If the file is missing, run:

```powershell
python evaluate_models.py
```

## Limitations

- Text-only predictions may miss contextual or behavioral signals.
- Dataset labels may contain noise.
- Test performance does not guarantee performance on unseen platforms or future reviews.
- Confidence values may not be calibrated probabilities.
- Agreement between models does not prove correctness.
- Evaluation metrics depend on the metric configuration.
- Live inference and evaluation should use identical preprocessing.

## Future Improvements

- Align preprocessing between training, evaluation, and live inference.
- Verify positive-label settings for evaluation metrics.
- Add per-class, macro, and weighted metrics.
- Add ROC-AUC and precision-recall curves.
- Improve dashboard filtering and pagination.
- Add automated tests for preprocessing and inference.
- Improve error handling and loading states.
- Evaluate models on external review datasets.
- Improve deployment scalability.

## Results Summary

On the reported test set of 6,059 reviews:

- **LSTM Accuracy:** 93.46%
- **DistilBERT Accuracy:** 97.95%
- **LSTM Correct Predictions:** 5,663
- **DistilBERT Correct Predictions:** 5,935
- **Model Agreement:** 93.20%

DistilBERT achieved stronger results than LSTM in the reported evaluation.

---

**Project:** Fake Review Detection System  
**Live Application:** https://fake-review-detection-system-vwmy8h9a2sfhccdcfzxeva.streamlit.app/  
**Repository:** https://github.com/nishanttripathi0061-hub/fake-review-detection-system