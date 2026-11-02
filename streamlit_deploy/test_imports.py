import streamlit as st

st.title("Import Diagnostic")

try:
    import torch
    st.success("PyTorch imported successfully")
except Exception as e:
    st.error(f"PyTorch error: {e}")

try:
    import tensorflow as tf
    st.success("TensorFlow imported successfully")
except Exception as e:
    st.error(f"TensorFlow error: {e}")

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    st.success("Transformers imported successfully")
except Exception as e:
    st.error(f"Transformers error: {e}")

st.write("All import tests completed.")