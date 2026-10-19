from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import hstack

from train import build_text, engineer


def main() -> None:
    parser = argparse.ArgumentParser(description="Score job postings for fraud risk")
    parser.add_argument("--input", required=True, help="CSV containing the training schema without fraudulent")
    parser.add_argument("--model", default="outputs/fake_job_detector.joblib")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()
    artifact = joblib.load(args.model)
    raw = pd.read_csv(args.input)
    data = engineer(raw)
    text = build_text(data)
    xs = artifact["structured_transformer"].transform(data)
    xw = artifact["word_vectorizer"].transform(text)
    xc = artifact["char_vectorizer"].transform(text)
    x = hstack([xw, xc, xs], format="csr")
    score = artifact["model"].predict_proba(x)[:, 1]
    result = raw.copy()
    result["fraud_risk_score"] = score
    result["fraud_prediction"] = (score >= artifact["threshold"]).astype(int)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} rows to {args.output}")


if __name__ == "__main__":
    main()
