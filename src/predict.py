"""Simple predict script that loads a model and predicts probabilities for a given CSV."""
import joblib
import pandas as pd
from .config import MODELS_DIR


def predict_csv(model_path, csv_path, proba=True):
    model = joblib.load(model_path)
    df = pd.read_csv(csv_path)
    X = df.select_dtypes(include=["number"]).copy()
    if proba and hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m src.predict <model_path> <csv_path>")
        sys.exit(1)
    model_path = sys.argv[1]
    csv_path = sys.argv[2]
    proba = True
    preds = predict_csv(model_path, csv_path, proba=proba)
    print(preds[:10])
