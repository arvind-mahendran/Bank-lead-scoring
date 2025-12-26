"""Training entrypoint: simple example using RandomForestClassifier."""
import joblib
from sklearn.ensemble import RandomForestClassifier
from .data_prep import train_test_split_df, simple_clean
from .config import MODELS_DIR
import pandas as pd


def train_model(df: pd.DataFrame, target: str):
    df = simple_clean(df)
    X_train, X_test, y_train, y_test = train_test_split_df(df, target)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "lead_scoring_model.joblib"
    joblib.dump(model, model_path)
    return model, model_path


if __name__ == "__main__":
    # Example: run `python -m src.train path/to/data.csv target_column`
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m src.train <csv_path> <target_column>")
        sys.exit(1)
    csv_path = sys.argv[1]
    target = sys.argv[2]
    df = pd.read_csv(csv_path)
    model, path = train_model(df, target)
    print(f"Saved model to {path}")
