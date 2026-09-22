"""
Train the resume role classifier.

Pipeline: TF-IDF (word unigrams+bigrams) -> LogisticRegression.
Trains on a stratified split of data/synthetic_resumes.csv, reports
accuracy/F1 on a held-out eval set, and saves the model to models/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "synthetic_resumes.csv"
MODEL_PATH = BASE_DIR / "models" / "resume_classifier.joblib"


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=2000, n_jobs=None)),
        ]
    )


def main(test_size: float = 0.2, seed: int = 42) -> dict:
    df = pd.read_csv(DATA_PATH)
    train, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["role_category"]
    )
    pipe = build_pipeline()
    pipe.fit(train["resume_text"], train["role_category"])

    preds = pipe.predict(test["resume_text"])
    accuracy = accuracy_score(test["role_category"], preds)
    print(f"Eval accuracy: {accuracy:.3f}  (n={len(test)})")
    print(classification_report(test["role_category"], preds))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    return {"accuracy": accuracy, "n_test": len(test)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(test_size=args.test_size, seed=args.seed)
