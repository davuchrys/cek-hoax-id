"""Train a TF-IDF + Logistic Regression hoax classifier.

Usage:
    python -m hoax_detector.train --data data/sample.csv --out models/model.joblib

The CSV needs two columns: `text` and `label` (1 = hoax, 0 = valid).
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .preprocess import clean_text


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def train(data_path: Path, out_path: Path, test_size: float = 0.25, seed: int = 42) -> dict:
    df = pd.read_csv(data_path).dropna(subset=["text", "label"])
    if df["label"].value_counts().min() < 4:
        raise ValueError(f"Need at least 4 rows per class to train; got {df['label'].value_counts().to_dict()}")
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"], df["label"].astype(int), test_size=test_size, random_state=seed, stratify=df["label"]
    )
    model = build_pipeline().fit(x_train, y_train)
    pred = model.predict(x_test)
    metrics = {
        "n_train": len(x_train),
        "n_test": len(x_test),
        "f1": float(f1_score(y_test, pred, average="macro")),
        "report": classification_report(y_test, pred, output_dict=True),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    out_path.with_name("metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/sample.csv"))
    p.add_argument("--out", type=Path, default=Path("models/model.joblib"))
    args = p.parse_args()
    m = train(args.data, args.out)
    print(f"Saved model to {args.out} | F1={m['f1']:.3f} (train={m['n_train']}, test={m['n_test']})")


if __name__ == "__main__":
    main()
