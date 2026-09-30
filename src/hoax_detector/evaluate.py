"""Score the trained model on a hand-collected set of real messages.

Usage:
    python -m hoax_detector.evaluate --data data/eval/messages.csv

CSV columns: `text`, `label` (1 = hoax, 0 = valid). The rows must NOT be in the training corpus.
Reports both the plain 0.5 cut-off and the three-way outcome the web UI shows
(hoax >= 0.65, valid <= 0.35, undecided in between).
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

HOAX_AT, VALID_AT = 0.65, 0.35


def evaluate(model, df: pd.DataFrame) -> dict:
    y = df["label"].astype(int)
    p = model.predict_proba(df["text"])[:, 1]
    pred = (p >= 0.5).astype(int)
    decided = (p >= HOAX_AT) | (p <= VALID_AT)
    ui_pred = (p >= HOAX_AT).astype(int)
    return {
        "p": p,
        "pred": pred,
        "auc": float(roc_auc_score(y, p)) if y.nunique() == 2 else None,
        "report": classification_report(y, pred, target_names=["valid", "hoax"], zero_division=0),
        "confusion": confusion_matrix(y, pred, labels=[0, 1]),
        "undecided_share": float((~decided).mean()),
        "decided_accuracy": float((ui_pred[decided] == y[decided]).mean()) if decided.any() else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/eval/messages.csv"))
    ap.add_argument("--model", type=Path, default=Path("models/model.joblib"))
    args = ap.parse_args()

    df = pd.read_csv(args.data).dropna(subset=["text", "label"])
    model = joblib.load(args.model)
    r = evaluate(model, df)

    print(f"{len(df)} messages ({int(df['label'].sum())} hoax, {int((df['label'] == 0).sum())} valid)\n")
    print(r["report"])
    print("Confusion matrix (rows = truth valid/hoax, cols = predicted valid/hoax):")
    print(r["confusion"], "\n")
    if r["auc"] is not None:
        print(f"ROC AUC (threshold-free): {r['auc']:.3f}")
    print(f"UI shows 'undecided' for {r['undecided_share']:.0%} of messages")
    if r["decided_accuracy"] is not None:
        print(f"Accuracy on the messages the UI decides: {r['decided_accuracy']:.0%}")

    wrong = df.assign(p=r["p"].round(3), pred=r["pred"])[r["pred"] != df["label"].astype(int)]
    if len(wrong):
        print(f"\nMisclassified ({len(wrong)}):")
        for _, row in wrong.iterrows():
            kind = "hoax->valid" if row["label"] == 1 else "valid->hoax"
            print(f"  [{kind} p={row['p']}] {str(row['text'])[:110]}")


if __name__ == "__main__":
    main()
