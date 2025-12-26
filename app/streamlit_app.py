import streamlit as st
import joblib
import pandas as pd
from src.config import MODELS_DIR

st.title("Bank Lead Scoring")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
model_file = st.text_input("Model path", str(MODELS_DIR / "lead_scoring_model.joblib"))

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head())
    if st.button("Predict"):
        model = joblib.load(model_file)
        X = df.select_dtypes(include=["number"]).copy()
        if hasattr(model, "predict_proba"):
            df["score"] = model.predict_proba(X)[:, 1]
        else:
            df["score"] = model.predict(X)
        st.dataframe(df.head())
