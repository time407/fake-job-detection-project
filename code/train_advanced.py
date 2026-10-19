from __future__ import annotations

import json
from pathlib import Path

import joblib
import xgboost as xgb
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.metrics import average_precision_score, precision_recall_curve

from train import build_text, choose_threshold, engineer, metric_dict


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "outputs"
    raw = pd.read_csv(root / "data" / "fake_job_postings.csv")
    data = engineer(raw)
    y = raw["fraudulent"].to_numpy()
    split = json.loads((out / "split_indices.json").read_text())
    tr = np.asarray(split["train"]); va = np.asarray(split["validation"]); te = np.asarray(split["test"])

    base = joblib.load(out / "fake_job_detector.joblib")
    text = build_text(data)
    xs_tr = base["structured_transformer"].transform(data.iloc[tr])
    xs_va = base["structured_transformer"].transform(data.iloc[va])
    xs_te = base["structured_transformer"].transform(data.iloc[te])
    xw_tr = base["word_vectorizer"].transform(text.iloc[tr])
    xw_va = base["word_vectorizer"].transform(text.iloc[va])
    xw_te = base["word_vectorizer"].transform(text.iloc[te])
    xc_tr = base["char_vectorizer"].transform(text.iloc[tr])
    xc_va = base["char_vectorizer"].transform(text.iloc[va])
    xc_te = base["char_vectorizer"].transform(text.iloc[te])
    xtr = hstack([xw_tr, xc_tr, xs_tr], format="csr")
    xva = hstack([xw_va, xc_va, xs_va], format="csr")
    xte = hstack([xw_te, xc_te, xs_te], format="csr")

    linear_va = base["model"].predict_proba(xva)[:, 1]
    linear_te = base["model"].predict_proba(xte)[:, 1]
    positive_weight = float((y[tr] == 0).sum() / (y[tr] == 1).sum())
    tree = xgb.XGBClassifier(
        objective="binary:logistic", n_estimators=1200, learning_rate=0.035,
        max_depth=5, min_child_weight=3, gamma=0.05,
        subsample=0.85, colsample_bytree=0.75, reg_alpha=0.2, reg_lambda=1.0,
        scale_pos_weight=positive_weight, random_state=42, n_jobs=-1,
        tree_method="hist", eval_metric="aucpr", early_stopping_rounds=100,
    )
    tree.fit(
        xs_tr, y[tr], eval_set=[(xs_va, y[va])], verbose=False,
    )
    tree_va = tree.predict_proba(xs_va)[:, 1]
    tree_te = tree.predict_proba(xs_te)[:, 1]

    # Validation-only search for the nonlinear/linear blend.
    candidates = []
    for weight_tree in np.linspace(0, 1, 101):
        score = weight_tree * tree_va + (1 - weight_tree) * linear_va
        candidates.append((average_precision_score(y[va], score), float(weight_tree)))
    _, weight_tree = max(candidates)
    blend_va = weight_tree * tree_va + (1 - weight_tree) * linear_va
    blend_te = weight_tree * tree_te + (1 - weight_tree) * linear_te
    threshold = choose_threshold(y[va], blend_va)

    results = {
        "architecture": "validation-weighted two-tower ensemble: XGBoost nonlinear structured tower plus word+character TF-IDF linear text tower",
        "xgboost_best_iteration": int(tree.best_iteration),
        "tree_weight": weight_tree,
        "linear_weight": 1 - weight_tree,
        "threshold": threshold,
        "linear_baseline_test": metric_dict(y[te], linear_te, base["threshold"]),
        "xgboost_test": metric_dict(y[te], tree_te, choose_threshold(y[va], tree_va)),
        "advanced_ensemble_validation": metric_dict(y[va], blend_va, threshold),
        "advanced_ensemble_test": metric_dict(y[te], blend_te, threshold),
    }
    (out / "advanced_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    joblib.dump({
        "base": base, "xgboost": tree, "tree_uses_structured_only": True, "tree_weight": weight_tree,
        "linear_weight": 1-weight_tree, "threshold": threshold,
    }, out / "fake_job_detector_advanced.joblib", compress=3)
    np.save(out / "advanced_test_scores.npy", blend_te)
    pred = (blend_te >= threshold).astype(int)
    error_rows = raw.iloc[te].copy()
    error_rows["score"] = blend_te
    error_rows["prediction"] = pred
    error_rows["error_type"] = np.where((y[te] == 1) & (pred == 0), "false_negative",
                                 np.where((y[te] == 0) & (pred == 1), "false_positive", "correct"))
    error_rows[error_rows.error_type != "correct"].sort_values("score", ascending=False).to_csv(
        out / "advanced_test_errors.csv", index=False
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
