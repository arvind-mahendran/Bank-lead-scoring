"""Small utilities for data loading, cleaning and splitting."""
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import RAW_DIR, PROCESSED_DIR


def load_csv(path):
    """Load CSV into DataFrame."""
    return pd.read_csv(path)


def simple_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop exact duplicates and reset index."""
    df = df.copy()
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def train_test_split_df(df, target, test_size=0.2, random_state=42):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
