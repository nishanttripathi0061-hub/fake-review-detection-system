# Fake Review Detection System

A machine-learning web application that classifies product reviews as
**Fake** or **Original** using two models: **LSTM** and **DistilBERT**.
The project includes a Flask review analyzer and an interactive
evaluation dashboard.

> **Caution:** Model predictions are estimates, not proof that a
> real-world review is fake or genuine.

## Table of Contents

-   [Overview](#overview)
-   [Objectives](#objectives)
-   [Technology Stack](#technology-stack)
-   [Workflow](#workflow)
-   [Dataset and Labels](#dataset-and-labels)
-   [Models and Inference](#models-and-inference)
-   [Evaluation Results](#evaluation-results)
-   [Confusion Matrices](#confusion-matrices)
-   [Model Agreement](#model-agreement)
-   [Web Application](#web-application)
-   [Project Structure](#project-structure)
-   [Setup and Run](#setup-and-run)
-   [Evaluation Predictions](#evaluation-predictions)
-   [Limitations](#limitations)
-   [Future Improvements](#future-improvements)

## Overview

The system compares two text-classification approaches on product
reviews: 1. **LSTM (Long Short-Term Memory):** a recurrent neural
network. 2. **DistilBERT:** a transformer-based sequence-classification
model.

Users can submit a review to the Flask web app and see both model
predictions. The evaluation dashboard visualizes aggregate metrics,
agreement, confusion-matrix counts, and individual test examples.

## Objectives

-   Build a binary fake-review classifier.
-   Train/evaluate LSTM and DistilBERT models.
-   Compare models on a held-out test set.
-   Provide a simple web interface for single-review inference.
-   Provide interactive charts and searchable test-prediction rows.
-   Save models locally for inference.

## Technology Stack

  Technology                  Use
  --------------------------- -------------------------------------------
  Python                      Main language
  TensorFlow / Keras          LSTM inference
  PyTorch                     DistilBERT inference
  Hugging Face Transformers   DistilBERT tokenizer/model
  Pandas                      CSV data handling
  NumPy                       Numerical operations
  scikit-learn                Evaluation metrics and confusion matrices
  Flask                       Web server and routes
  HTML / CSS                  Page structure and styling
  JavaScript                  Search, filters, charts
  Chart.js                    Dashboard charts
  VS Code / PowerShell        Development environment

### Reported environment

Virtual environment: `D:\\fake-review-env`\
Project directory: `D:\\fake review detection system`

The package list reported during development included Flask 3.1.3,
TensorFlow 2.21.0, PyTorch 2.11.0+cu128, Transformers 5.21.0, Pandas
3.1.0, NumPy 2.5.4, and scikit-learn 1.9.1. Versions can change; verify
compatibility when recreating the environment.

## Workflow

1.  Load the raw review dataset.
2.  Split data into training, validation, and test sets with a reported
    stratified split of 70% / 15% / 15%.
3.  Prepare review text for model input.
4.  Train/evaluate the LSTM model and save its model/tokenizer.
5.  Train/evaluate or load the locally saved DistilBERT
    sequence-classification model.
6.  Run both models on the held-out test set.
7.  Save predictions to `data/test_predictions.csv`.
8.  Serve the analyzer and dashboard through Flask.

## Dataset and Labels

  Item                         Details
  ---------------------------- -------------------------------------
  Raw dataset                  `data/fake reviews dataset.csv`
  Training set                 `data/train.csv`
  Validation set               `data/validation.csv`
  Test set                     `data/test.csv`
  Evaluation output            `data/test_predictions.csv`
  Reported full dataset size   Approximately 40,432 rows
  Reported test size           6,059 reviews
  Fake label                   `CG`, numeric ID `0`
  Original label               `OR`, numeric ID `1`
  Split                        70% train, 15% validation, 15% test
  Strategy                     Stratified split

The evaluation script reportedly selects `clean_text` if available,
otherwise `text_`. Confirm the exact cleaning steps in the current
notebook/script before claiming specific preprocessing details.

## Models and Inference

### LSTM

-   Saved model: `models/best_lstm_model.keras`
-   Saved tokenizer: `models/lstm_tokenizer.pkl`
-   Framework: TensorFlow/Keras
-   Input length: 250 tokens
-   Padding/truncation: post-padding and post-truncation
-   Sigmoid score is interpreted as class 1 = Original.
-   Threshold: score \>= 0.5 predicts Original; otherwise Fake.
-   Displayed confidence is the larger of the score and its complement.

### DistilBERT

-   Local model directory: `models/distilbert-fake-review-model/`
-   Framework: PyTorch and Hugging Face Transformers
-   Maximum token length: 256
-   Class mapping: 0 = Fake, 1 = Original
-   Uses softmax probabilities and selects the highest-probability
    class.
-   Device selection: CUDA when available, otherwise CPU.

### Comparing predictions

The web app sends the review to both models. Matching labels produce an
agreement message; different labels produce a disagreement warning.
Agreement does not guarantee correctness.

## Evaluation Results

Reported results on the held-out test set of **6,059 reviews**:

  Metric                      LSTM   DistilBERT         DistilBERT − LSTM
  ----------------------- -------- ------------ -------------------------
  Accuracy                  93.46%       97.95%   +4.49 percentage points
  Precision                 93.33%       99.06%   +5.73 percentage points
  Recall                    93.63%       96.83%   +3.20 percentage points
  F1-score                  93.48%       97.93%   +4.45 percentage points
  Correct predictions        5,663        5,935                      +272
  Incorrect predictions        396          124                      −272

**Result:** DistilBERT outperformed LSTM on the reported aggregate
metrics, with 272 more correct predictions and 272 fewer incorrect
predictions on this test set.

### Metric interpretation caveat

Accuracy is the fraction of all examples predicted correctly. Precision,
recall, and F1 depend on the evaluation code's `average` and `pos_label`
settings. The recorded values are reproduced from the reported
evaluation. With the mapping above, class `1` is Original; check
`evaluate_models.py` before describing precision/recall/F1 as
specifically Fake-class metrics.

## Confusion Matrices

Rows are actual labels and columns are predicted labels. Class order:
Fake (0), Original (1).

### LSTM

  Actual  Predicted     Fake (0)   Original (1)
  ------------------- ---------- --------------
  Fake (0)                 2,824            203
  Original (1)               193          2,839

-   Correct: 2,824 + 2,839 = **5,663**
-   Incorrect: 203 + 193 = **396**

### DistilBERT

  Actual  Predicted     Fake (0)   Original (1)
  ------------------- ---------- --------------
  Fake (0)                 2,999             28
  Original (1)                96          2,936

-   Correct: 2,999 + 2,936 = **5,935**
-   Incorrect: 28 + 96 = **124**

DistilBERT made fewer errors in both off-diagonal categories on this
test set: - Fake predicted as Original: 28 vs. 203 for LSTM. - Original
predicted as Fake: 96 vs. 193 for LSTM.

## Model Agreement

  Measure                      Result
  ------------------- ---------------
  Both models agree     5,647 reviews
  Agreement rate               93.20%
  Models disagree         412 reviews
  Disagreement rate             6.80%

Agreement is computed by comparing `lstm_prediction_id` with
`distilbert_prediction_id`. It does not establish that either prediction
is correct.

## Web Application

### Review Analyzer

Local URL: `http://127.0.0.1:5000/`

Features: - Enter/paste a review. - See LSTM prediction and
confidence. - See DistilBERT prediction and confidence. - See whether
the two models agree.

### Evaluation Dashboard

Local URL: `http://127.0.0.1:5000/dashboard`

Features: - Test-set count and accuracy cards. - Metrics comparison
chart. - Agreement/disagreement doughnut chart. - Confusion-matrix count
charts. - Searchable review table. - Filters for disagreement, at least
one wrong prediction, both wrong, and both correct. - Visible-row limit
controls.

The current dashboard table loads up to the first **1,000 rows** for
browser-side filtering, while aggregate metrics are calculated from the
full CSV. Chart.js is loaded from a CDN and therefore needs internet
access unless hosted locally.

## Project Structure

Core structure used during development:

``` text
fake review detection system/
├── app.py
├── evaluate_models.py
├── data/
│   ├── fake reviews dataset.csv
│   ├── train.csv
│   ├── validation.csv
│   ├── test.csv
│   └── test_predictions.csv
├── models/
│   ├── best_lstm_model.keras
│   ├── lstm_tokenizer.pkl
│   └── distilbert-fake-review-model/
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_lstm_model.ipynb
│   ├── 04_distilbert_model.ipynb
│   └── 05_model_comparison.ipynb
└── templates/
    ├── index.html
    └── dashboard.html
```

The frontend was also being reorganized into `frontend/html`,
`frontend/css`, and `frontend/js`. If that change is completed, update
this tree and configure Flask to use those template/static paths. Flask
defaults to `templates/` and `static/`; moving files without
configuration can break the routes and assets.

## Setup and Run

Open the project folder in VS Code, then open **Terminal → New
Terminal**.

``` powershell
cd "D:\\fake review detection system"
& "D:\\fake-review-env\\Scripts\\Activate.ps1"
python app.py
```

Keep the terminal open while the server runs. Open: - Analyzer:
`http://127.0.0.1:5000/` - Dashboard: `http://127.0.0.1:5000/dashboard`

Stop the server with `Ctrl + C`. The `127.0.0.1` address is local to the
machine running Flask, not a public deployment link.

### Check/install dependencies

``` powershell
python -m pip list
python -m pip freeze > requirements.txt
```

Only if required packages are missing:

``` powershell
python -m pip install flask pandas numpy scikit-learn tensorflow torch transformers
```

Avoid changing working package versions without checking compatibility.

## Evaluation Predictions

The dashboard requires `data/test_predictions.csv`. If it is missing,
run this command from the activated environment and project root:

``` powershell
python evaluate_models.py
```

The output was reported to contain columns such as: -
`actual_label_id` - `lstm_prediction_id` - `distilbert_prediction_id` -
`models_agree` - `lstm_prediction` - `distilbert_prediction` -
`actual_label`

The review table looks for review text in `clean_text` or `text_`.
Confirm the actual CSV column names if the text column appears blank.

## Limitations

-   Text-only predictions may miss contextual or behavioral signals.
-   Dataset labels may contain noise.
-   Test performance does not guarantee performance on new platforms,
    products, languages, or future reviews.
-   Confidence values may not be calibrated probabilities.
-   Agreement does not prove correctness.
-   The dashboard table currently samples up to 1,000 rows.
-   Live inference and evaluation should use identical preprocessing;
    the evaluation script may use cleaned text while live inference may
    receive raw text.
-   Metrics depend on the metric configuration in `evaluate_models.py`.

## Future Improvements

-   Document Python version and pin compatible package versions.
-   Align preprocessing between evaluation and live inference.
-   Verify positive-label settings for precision, recall, and F1.
-   Add per-class and macro/weighted metrics.
-   Add ROC-AUC and precision-recall curves where appropriate.
-   Use a 2×2 heatmap for confusion matrices.
-   Add pagination or server-side filtering for all test rows.
-   Add automated tests for preprocessing, model loading, routes, and
    dashboard data.
-   Add loading states and robust inference error handling.
-   Host Chart.js locally for offline use.
-   Evaluate on an external dataset and consider deployment behind a
    production server.

## Troubleshooting

### Pylance says `Import "torch" could not be resolved`

Select the correct interpreter in VS Code: 1. Press `Ctrl + Shift + P`.
2. Choose **Python: Select Interpreter**. 3. Select
`D:\\fake-review-env\\Scripts\\python.exe`. 4. Reload the VS Code window
if needed.

Verify the import:

``` powershell
& "D:\\fake-review-env\\Scripts\\python.exe" -c "import torch; print(torch.__version__)"
```

### `ModuleNotFoundError`

Activate the correct environment, then install only the missing package:

``` powershell
python -m pip install PACKAGE_NAME
```

### Dashboard says the CSV is missing

Run `python evaluate_models.py` from the project root and check
`data/test_predictions.csv`.

### Port 5000 is already in use

Stop the previous Flask process with `Ctrl + C`, then run
`python app.py` again.

### Charts are blank

Check internet connectivity because Chart.js is loaded from a CDN.
Inspect the browser console for JavaScript errors.

## Results Summary

On the reported test set of 6,059 reviews: - **LSTM accuracy:** 93.46% -
**DistilBERT accuracy:** 97.95% - **LSTM correct predictions:** 5,663 -
**DistilBERT correct predictions:** 5,935 - **Agreement:** 5,647 reviews
(93.20%) - **Disagreement:** 412 reviews (6.80%)

DistilBERT was the stronger model in the reported comparison. Before
submitting or publishing this project, rerun the current evaluation
script and verify the scores, metric settings, file paths, and final
frontend folder structure against the actual code.
