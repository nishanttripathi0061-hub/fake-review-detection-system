from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import torch
import tensorflow as tf

from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

LSTM_REPO_ID = "nishant-works/fake-review-lstm"
DISTILBERT_REPO_ID = "nishant-works/fake-review-distilbert"

PREDICTIONS_PATH = BASE_DIR.parent / "data" / "test_predictions.csv"

DEVICE = torch.device("cpu")
torch.set_num_threads(2)

st.set_page_config(
    page_title="Fake Review Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# 2. GLOBAL STYLING
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #090d16;
        --panel: rgba(20, 29, 46, .78);
        --panel-strong: #151e30;
        --border: rgba(148, 163, 184, .17);
        --muted: #94a3b8;
        --text: #edf3ff;
        --blue: #60a5fa;
        --purple: #a78bfa;
        --green: #34d399;
        --red: #fb7185;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 85% 0%, rgba(59,130,246,.13), transparent 34%),
            radial-gradient(ellipse at 10% 20%, rgba(124,58,237,.10), transparent 32%),
            var(--bg);
        color: var(--text);
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #151d2d 0%, #0e1523 100%);
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    section[data-testid="stSidebar"] h1 {
        font-size: 1.35rem;
        line-height: 1.35;
        color: #f8fafc;
    }

    h1, h2, h3 {
        color: var(--text) !important;
        letter-spacing: -.035em;
        font-weight: 750 !important;
    }

    h1 {
        background: linear-gradient(95deg, #f8fbff 0%, #93c5fd 48%, #c4b5fd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    p, label, .stMarkdown, [data-testid="stCaptionContainer"] {
        color: #cbd5e1;
    }

    .hero-kicker {
        color: #93c5fd;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .16em;
        text-transform: uppercase;
        margin-bottom: .5rem;
    }

    .hero-subtitle {
        color: #a9b8ce;
        font-size: 1rem;
        margin-top: -.25rem;
        margin-bottom: 1.5rem;
    }

    .section-heading {
        color: #f1f5f9;
        font-size: 1.15rem;
        font-weight: 750;
        margin: 1.2rem 0 .9rem;
    }

    .glass-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, .82), rgba(15, 23, 42, .88));
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.25rem 1.35rem;
        box-shadow: 0 12px 35px rgba(0,0,0,.16);
        backdrop-filter: blur(14px);
        height: 100%;
    }

    .metric-label {
        color: #9fb0c8;
        font-size: .82rem;
        font-weight: 600;
        margin-bottom: .55rem;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -.04em;
    }

    .metric-note {
        color: #8292aa;
        font-size: .75rem;
        margin-top: .35rem;
    }

    .model-card {
        background: linear-gradient(145deg, rgba(25, 37, 57, .94), rgba(13, 20, 33, .96));
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.35rem;
        min-height: 235px;
        box-shadow: 0 14px 35px rgba(0,0,0,.18);
    }

    .model-name {
        font-size: 1.05rem;
        font-weight: 750;
        color: #eaf2ff;
        margin-bottom: 1rem;
    }

    .prediction-badge {
        display: inline-block;
        border-radius: 999px;
        padding: .45rem .85rem;
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .04em;
        margin-bottom: 1.05rem;
    }

    .fake-badge {
        color: #fecdd3;
        background: rgba(244,63,94,.16);
        border: 1px solid rgba(251,113,133,.35);
    }

    .original-badge {
        color: #a7f3d0;
        background: rgba(16,185,129,.14);
        border: 1px solid rgba(52,211,153,.35);
    }

    .confidence-number {
        font-size: 1.65rem;
        font-weight: 800;
        color: #f8fafc;
        margin: .15rem 0 .65rem;
    }

    .confidence-track {
        height: 8px;
        width: 100%;
        background: rgba(148,163,184,.18);
        border-radius: 99px;
        overflow: hidden;
    }

    .confidence-fill {
        height: 100%;
        border-radius: 99px;
        background: linear-gradient(90deg, #3b82f6, #a78bfa);
    }

    .comparison {
        border-radius: 14px;
        padding: 1rem 1.2rem;
        border: 1px solid var(--border);
        background: rgba(30,41,59,.55);
        color: #e2e8f0;
        font-weight: 650;
    }

    .comparison.agree {
        border-color: rgba(52,211,153,.3);
        background: rgba(16,185,129,.10);
    }

    .comparison.disagree {
        border-color: rgba(251,191,36,.32);
        background: rgba(245,158,11,.10);
    }

    .info-note {
        border: 1px solid rgba(96,165,250,.22);
        background: rgba(37,99,235,.08);
        color: #b9c9df;
        border-radius: 12px;
        padding: .85rem 1rem;
        font-size: .82rem;
        line-height: 1.55;
    }

    .stTextArea textarea {
        background: rgba(15, 23, 42, .86) !important;
        color: #f8fafc !important;
        border: 1px solid rgba(148,163,184,.25) !important;
        border-radius: 14px !important;
        padding: 1rem !important;
        font-size: .96rem !important;
    }

    .stTextArea textarea:focus {
        border-color: #60a5fa !important;
        box-shadow: 0 0 0 2px rgba(96,165,250,.13) !important;
    }

    .stButton > button, .stDownloadButton > button {
        border: 0 !important;
        border-radius: 10px !important;
        color: white !important;
        font-weight: 700 !important;
        background: linear-gradient(100deg, #2563eb, #7c3aed) !important;
        padding: .62rem 1.25rem !important;
        transition: transform .2s ease, box-shadow .2s ease;
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(59,130,246,.28);
    }

    div[data-testid="stDataFrame"], div[data-testid="stTable"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    [data-testid="stMetric"] {
        background: rgba(30,41,59,.55);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem;
    }

    [data-testid="stMetricLabel"] {
        color: #9fb0c8 !important;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-weight: 800;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    hr {
        border-color: var(--border) !important;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 3. MODEL LOADING
# =========================================================

@st.cache_resource(show_spinner=False)
def load_lstm():

    model_path = hf_hub_download(
        repo_id=LSTM_REPO_ID,
        filename="best_lstm_model.keras",
    )

    tokenizer_path = hf_hub_download(
        repo_id=LSTM_REPO_ID,
        filename="lstm_tokenizer.pkl",
    )

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    with open(tokenizer_path, "rb") as file:
        tokenizer = pickle.load(file)

    return model, tokenizer


@st.cache_resource(show_spinner=False)
def load_distilbert():

    model_dir = snapshot_download(
        repo_id=DISTILBERT_REPO_ID
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir
    ).to(DEVICE)

    model.eval()

    return tokenizer, model


@st.cache_resource(show_spinner=False)
def load_all_models():

    lstm_model, lstm_tokenizer = load_lstm()

    bert_tokenizer, bert_model = load_distilbert()

    return (
        lstm_model,
        lstm_tokenizer,
        bert_tokenizer,
        bert_model,
    )


# =========================================================
# 4. PREDICTION FUNCTIONS
# =========================================================

def predict_lstm(review, model, tokenizer):

    sequence = tokenizer.texts_to_sequences([review])

    padded = tf.keras.utils.pad_sequences(
        sequence,
        maxlen=250,
        padding="post",
        truncating="post",
    )

    score = float(
        model.predict(padded, verbose=0)[0][0]
    )

    label = "Original" if score >= 0.5 else "Fake"

    confidence = max(score, 1 - score) * 100

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "score": round(score, 4),
    }


def predict_distilbert(review, tokenizer, model):

    inputs = tokenizer(
        review,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1,
        )[0]

        predicted_class = int(
            torch.argmax(probabilities).item()
        )

    label = "Original" if predicted_class == 1 else "Fake"

    confidence = float(
        probabilities[predicted_class].item()
    ) * 100

    original_score = float(probabilities[1].item())

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "score": round(original_score, 4),
    }


# =========================================================
# 5. REUSABLE UI HELPERS
# =========================================================

def render_metric_card(label, value, note=""):

    st.markdown(
        f"""
        <div class="glass-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_card(model_name, result):

    is_fake = result["label"] == "Fake"

    badge_class = "fake-badge" if is_fake else "original-badge"

    badge_text = "FAKE REVIEW" if is_fake else "ORIGINAL REVIEW"

    st.markdown(
        f"""
        <div class="model-card">
            <div class="model-name">{model_name}</div>
            <div class="prediction-badge {badge_class}">{badge_text}</div>
            <div class="metric-label">Model confidence</div>
            <div class="confidence-number">{result['confidence']:.2f}%</div>
            <div class="confidence-track">
                <div class="confidence-fill" style="width:{result['confidence']}%"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def calculate_metrics(actual, predicted):

    return {
        "Accuracy": round(
            accuracy_score(actual, predicted) * 100, 2
        ),

        "Precision": round(
            precision_score(
                actual,
                predicted,
                zero_division=0,
            ) * 100, 2
        ),

        "Recall": round(
            recall_score(
                actual,
                predicted,
                zero_division=0,
            ) * 100, 2
        ),

        "F1 Score": round(
            f1_score(
                actual,
                predicted,
                zero_division=0,
            ) * 100, 2
        ),

        "Confusion Matrix": confusion_matrix(
            actual,
            predicted,
            labels=[0, 1],
        ),
    }


# =========================================================
# 6. SIDEBAR NAVIGATION
# =========================================================

st.sidebar.markdown("## 🔍 Fake Review\n## Detection")

st.sidebar.caption("AI-powered review analysis")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    ["Review Analyzer", "Model Dashboard"],
    label_visibility="visible",
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <div class="info-note">
    <b>Models</b><br>
    LSTM · DistilBERT<br><br>
    <b>Classes</b><br>
    Fake (CG) · Original (OR)
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.caption("Fake Review Detection System")


# =========================================================
# 7. REVIEW ANALYZER
# =========================================================

if page == "Review Analyzer":

    st.markdown(
        '<div class="hero-kicker">AI REVIEW INTELLIGENCE</div>',
        unsafe_allow_html=True,
    )

    st.title("Fake Review Detection System")

    st.markdown(
        '<div class="hero-subtitle">Analyze a product review using two deep-learning models: LSTM and DistilBERT.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-heading">Enter product review</div>',
        unsafe_allow_html=True,
    )

    review = st.text_area(
        "Review text",
        height=165,
        placeholder="Write or paste a product review here...",
        label_visibility="collapsed",
        key="review_input",
    )

    word_count = len(review.split()) if review.strip() else 0

    st.caption(f"{len(review)} characters · {word_count} words")

    analyze_col, clear_col = st.columns([1, 5])

    with analyze_col:

        analyze_clicked = st.button(
            "✨ Analyze Review",
            type="primary",
            use_container_width=True,
        )

    with clear_col:

        st.caption("Both models analyze the same review independently.")

    if analyze_clicked:

        if not review.strip():

            st.warning("Please enter a review before analyzing.")

        else:

            try:

                with st.spinner("Loading models and analyzing your review..."):

                    (
                        lstm_model,
                        lstm_tokenizer,
                        bert_tokenizer,
                        bert_model,
                    ) = load_all_models()

                    lstm_result = predict_lstm(
                        review,
                        lstm_model,
                        lstm_tokenizer,
                    )

                    bert_result = predict_distilbert(
                        review,
                        bert_tokenizer,
                        bert_model,
                    )

                st.session_state["last_review"] = review

                st.session_state["last_lstm_result"] = lstm_result

                st.session_state["last_bert_result"] = bert_result

            except Exception as error:

                st.error(
                    "Prediction failed. Check the model download, environment, and network connection."
                )

                with st.expander("Technical error details"):

                    st.exception(error)

    if "last_lstm_result" in st.session_state and "last_bert_result" in st.session_state:

        st.divider()

        st.markdown(
            '<div class="section-heading">Prediction results</div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2, gap="large")

        with col1:

            render_prediction_card(
                "LSTM Model",
                st.session_state["last_lstm_result"],
            )

        with col2:

            render_prediction_card(
                "DistilBERT Model",
                st.session_state["last_bert_result"],
            )

        lstm_result = st.session_state["last_lstm_result"]

        bert_result = st.session_state["last_bert_result"]

        st.markdown(
            '<div class="section-heading">Model comparison</div>',
            unsafe_allow_html=True,
        )

        if lstm_result["label"] == bert_result["label"]:

            comparison_text = f"✓ Both models agree: {lstm_result['label']} Review"

            comparison_class = "agree"

        else:

            comparison_text = "⚠ Models disagree — review the predictions carefully."

            comparison_class = "disagree"

        st.markdown(
            f'<div class="comparison {comparison_class}">{comparison_text}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="info-note" style="margin-top:1rem;">Confidence is the selected class probability reported by each model. A model prediction is an automated estimate, not proof that a review is genuine or fabricated.</div>',
            unsafe_allow_html=True,
        )


# =========================================================
# 8. MODEL DASHBOARD
# =========================================================

elif page == "Model Dashboard":

    st.markdown(
        '<div class="hero-kicker">MODEL PERFORMANCE & EVALUATION</div>',
        unsafe_allow_html=True,
    )

    st.title("Model Evaluation Dashboard")

    st.markdown(
        '<div class="hero-subtitle">A comparative view of LSTM and DistilBERT performance on the held-out test predictions.</div>',
        unsafe_allow_html=True,
    )

    if not PREDICTIONS_PATH.exists():

        st.error("test_predictions.csv not found. Make sure it exists inside the data folder.")

        st.stop()

    try:

        df = pd.read_csv(PREDICTIONS_PATH)

    except Exception as error:

        st.error(f"Could not read test_predictions.csv: {error}")

        st.stop()

    required_columns = [
        "actual_label_id",
        "lstm_prediction_id",
        "distilbert_prediction_id",
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:

        st.error(
            f"Required prediction columns are missing: {', '.join(missing)}"
        )

        st.stop()

    df = df.dropna(subset=required_columns).copy()

    y_true = df["actual_label_id"].astype(int).to_numpy()

    lstm_pred = df["lstm_prediction_id"].astype(int).to_numpy()

    bert_pred = df["distilbert_prediction_id"].astype(int).to_numpy()

    if not (
        set(np.unique(y_true)).issubset({0, 1})
        and set(np.unique(lstm_pred)).issubset({0, 1})
        and set(np.unique(bert_pred)).issubset({0, 1})
    ):

        st.error("Unexpected class IDs found. Expected 0=Fake and 1=Original.")

        st.stop()

    lstm_metrics = calculate_metrics(y_true, lstm_pred)

    bert_metrics = calculate_metrics(y_true, bert_pred)

    agreement = lstm_pred == bert_pred

    agreement_count = int(agreement.sum())

    disagreement_count = int((~agreement).sum())

    agreement_pct = agreement.mean() * 100 if len(agreement) else 0

    disagreement_pct = (~agreement).mean() * 100 if len(agreement) else 0

    # KPI CARDS

    k1, k2, k3, k4 = st.columns(4, gap="medium")

    with k1:

        render_metric_card(
            "TEST REVIEWS",
            f"{len(df):,}",
            "Held-out prediction records",
        )

    with k2:

        render_metric_card(
            "LSTM ACCURACY",
            f"{lstm_metrics['Accuracy']:.2f}%",
            "Test-set accuracy",
        )

    with k3:

        render_metric_card(
            "DISTILBERT ACCURACY",
            f"{bert_metrics['Accuracy']:.2f}%",
            "Test-set accuracy",
        )

    with k4:

        render_metric_card(
            "MODEL AGREEMENT",
            f"{agreement_pct:.2f}%",
            f"{agreement_count:,} matching predictions",
        )

    k5, k6 = st.columns(2, gap="medium")

    with k5:

        render_metric_card(
            "AGREEMENTS",
            f"{agreement_count:,}",
            "Both models predicted the same class",
        )

    with k6:

        render_metric_card(
            "DISAGREEMENTS",
            f"{disagreement_count:,}",
            f"{disagreement_pct:.2f}% of test records",
        )

    st.divider()

    st.markdown(
        '<div class="section-heading">Model performance comparison</div>',
        unsafe_allow_html=True,
    )

    metrics_df = pd.DataFrame({

        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],

        "LSTM": [
            lstm_metrics["Accuracy"],
            lstm_metrics["Precision"],
            lstm_metrics["Recall"],
            lstm_metrics["F1 Score"],
        ],

        "DistilBERT": [
            bert_metrics["Accuracy"],
            bert_metrics["Precision"],
            bert_metrics["Recall"],
            bert_metrics["F1 Score"],
        ],
    })

    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True,
    )

    chart_df = metrics_df.set_index("Metric")

    st.bar_chart(
        chart_df,
        use_container_width=True,
        height=320,
    )

    st.markdown(
        '<div class="info-note">Precision, recall, and F1 are calculated using scikit-learn defaults, where class ID 1 (Original) is treated as the positive class. Class mapping: 0 = Fake (CG), 1 = Original (OR).</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        '<div class="section-heading">Confusion matrices</div>',
        unsafe_allow_html=True,
    )

    cm1, cm2 = st.columns(2, gap="large")

    with cm1:

        st.markdown("#### LSTM")

        lstm_cm_df = pd.DataFrame(
            lstm_metrics["Confusion Matrix"],
            index=["Actual Fake", "Actual Original"],
            columns=["Predicted Fake", "Predicted Original"],
        )

        st.dataframe(
            lstm_cm_df,
            use_container_width=True,
        )

    with cm2:

        st.markdown("#### DistilBERT")

        bert_cm_df = pd.DataFrame(
            bert_metrics["Confusion Matrix"],
            index=["Actual Fake", "Actual Original"],
            columns=["Predicted Fake", "Predicted Original"],
        )

        st.dataframe(
            bert_cm_df,
            use_container_width=True,
        )

    st.divider()

    st.markdown(
        '<div class="section-heading">Test prediction records</div>',
        unsafe_allow_html=True,
    )

    result_df = df.copy()

    result_df["Actual"] = np.where(
        y_true == 1,
        "Original",
        "Fake",
    )

    result_df["LSTM Prediction"] = np.where(
        lstm_pred == 1,
        "Original",
        "Fake",
    )

    result_df["DistilBERT Prediction"] = np.where(
        bert_pred == 1,
        "Original",
        "Fake",
    )

    result_df["Models Agree"] = agreement

    result_df["At Least One Wrong"] = (
        (lstm_pred != y_true) | (bert_pred != y_true)
    )

    result_df["Both Wrong"] = (
        (lstm_pred != y_true) & (bert_pred != y_true)
    )

    result_df["Both Correct"] = (
        (lstm_pred == y_true) & (bert_pred == y_true)
    )

    filter_col, search_col, limit_col = st.columns([1.25, 2, 1])

    with filter_col:

        record_filter = st.selectbox(
            "Filter records",
            [
                "All",
                "Disagree",
                "At Least One Wrong",
                "Both Wrong",
                "Both Correct",
            ],
        )

    with search_col:

        search_text = st.text_input(
            "Search review text or category",
            placeholder="Type to search...",
        )

    with limit_col:

        row_limit = st.selectbox(
            "Rows to show",
            [50, 100, 250, 1000],
            index=1,
        )

    filtered_df = result_df.copy()

    if record_filter == "Disagree":

        filtered_df = filtered_df[
            ~filtered_df["Models Agree"]
        ]

    elif record_filter == "At Least One Wrong":

        filtered_df = filtered_df[
            filtered_df["At Least One Wrong"]
        ]

    elif record_filter == "Both Wrong":

        filtered_df = filtered_df[
            filtered_df["Both Wrong"]
        ]

    elif record_filter == "Both Correct":

        filtered_df = filtered_df[
            filtered_df["Both Correct"]
        ]

    if search_text.strip():

        searchable_cols = [
            column for column in ["category", "text_", "clean_text"]
            if column in filtered_df.columns
        ]

        if searchable_cols:

            search_mask = pd.Series(
                False,
                index=filtered_df.index,
            )

            for column in searchable_cols:

                search_mask |= filtered_df[column].astype(str).str.contains(
                    search_text,
                    case=False,
                    na=False,
                    regex=False,
                )

            filtered_df = filtered_df[search_mask]

    st.caption(
        f"Showing {min(row_limit, len(filtered_df)):,} of {len(filtered_df):,} matching records."
    )

    display_columns = [
        column for column in [
            "category",
            "rating",
            "label",
            "text_",
            "clean_text",
            "Actual",
            "LSTM Prediction",
            "DistilBERT Prediction",
            "Models Agree",
        ]
        if column in filtered_df.columns
    ]

    st.dataframe(
        filtered_df[display_columns].head(row_limit),
        use_container_width=True,
        height=430,
        hide_index=True,
    )

    csv_data = filtered_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇ Download filtered records (CSV)",
        data=csv_data,
        file_name="fake_review_filtered_predictions.csv",
        mime="text/csv",
    )

    st.markdown(
        '<div class="info-note" style="margin-top:1rem;">Dataset labels are the reference labels used for evaluation; they are not independent proof of the real-world authenticity of an individual review.</div>',
        unsafe_allow_html=True,
    )


# =========================================================
# 9. FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#718096;font-size:.76rem;padding:.5rem 0 0;">
        Fake Review Detection System · LSTM + DistilBERT
    </div>
    """,
    unsafe_allow_html=True,
)