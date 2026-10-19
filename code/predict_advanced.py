from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd
from scipy.sparse import hstack

from train import build_text, engineer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--model", default="outputs/fake_job_detector_advanced.joblib")
    p.add_argument("--output", default="advanced_predictions.csv")
    a = p.parse_args()
    artifact = joblib.load(a.model); base = artifact["base"]
    raw = pd.read_csv(a.input); data = engineer(raw); text = build_text(data)
    xs = base["structured_transformer"].transform(data)
    xw = base["word_vectorizer"].transform(text)
    xc = base["char_vectorizer"].transform(text)
    x = hstack([xw, xc, xs], format="csr")
    linear = base["model"].predict_proba(x)[:, 1]
    tree = artifact["xgboost"].predict_proba(xs)[:, 1]
    score = artifact["tree_weight"] * tree + artifact["linear_weight"] * linear
    result = raw.copy(); result["fraud_risk_score"] = score
    result["fraud_prediction"] = (score >= artifact["threshold"]).astype(int)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True); result.to_csv(a.output, index=False)
    print(f"Wrote {len(result)} rows to {a.output}")


if __name__ == "__main__": main()
