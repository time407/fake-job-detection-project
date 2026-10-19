from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TEXT_COLUMNS = ["title", "company_profile", "description", "requirements", "benefits"]
CATEGORICAL_COLUMNS = [
    "department", "employment_type", "required_experience", "required_education",
    "industry", "function", "country", "region", "city",
]
BINARY_COLUMNS = ["telecommuting", "has_company_logo", "has_questions"]


def normalize_group_text(value: str) -> str:
    value = re.sub(r"\W+", " ", value.lower())
    return re.sub(r"\s+", " ", value).strip()


def build_text(df: pd.DataFrame) -> pd.Series:
    # Field markers preserve which section each token came from.
    parts = []
    for col in TEXT_COLUMNS:
        clean = df[col].fillna("").astype(str).str.replace(r"\s+", " ", regex=True)
        parts.append(" __" + col.upper() + "__ " + clean)
    return pd.concat(parts, axis=1).agg(" ".join, axis=1)


def parse_salary(value: object) -> tuple[float, float, float, float]:
    if pd.isna(value) or not str(value).strip():
        return np.nan, np.nan, np.nan, 1.0
    nums = []
    for token in re.findall(r"\d+(?:\.\d+)?", str(value).replace(",", "")):
        try:
            nums.append(float(token))
        except ValueError:
            pass
    if not nums:
        return np.nan, np.nan, np.nan, 0.0
    low, high = nums[0], nums[-1]
    return low, high, max(0.0, high - low), 0.0


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    location = out["location"].fillna("").astype(str).str.split(",", n=2, expand=True)
    out["country"] = location.get(0, "").replace("", "__MISSING__").str.strip()
    out["region"] = location.get(1, "").fillna("__MISSING__").replace("", "__MISSING__").str.strip()
    out["city"] = location.get(2, "").fillna("__MISSING__").replace("", "__MISSING__").str.strip()
    salary = np.array([parse_salary(x) for x in out["salary_range"]], dtype=float)
    out[["salary_low", "salary_high", "salary_width", "salary_missing"]] = salary
    for col in TEXT_COLUMNS:
        s = out[col].fillna("").astype(str)
        out[f"{col}_chars"] = s.str.len()
        out[f"{col}_words"] = s.str.split().str.len()
        out[f"{col}_missing"] = out[col].isna().astype(int)
        out[f"{col}_url_count"] = s.str.count(r"https?://|www\.")
        out[f"{col}_email_count"] = s.str.count(r"[\w.+-]+@[\w.-]+")
        out[f"{col}_digit_ratio"] = s.str.count(r"\d") / s.str.len().clip(lower=1)
        out[f"{col}_upper_ratio"] = s.str.count(r"[A-Z]") / s.str.len().clip(lower=1)
        out[f"{col}_bang_count"] = s.str.count("!")
    for col in CATEGORICAL_COLUMNS:
        out[col] = out[col].fillna("__MISSING__").astype(str)
    return out


def split_groups(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    combined = df[TEXT_COLUMNS].fillna("").astype(str).agg(" ".join, axis=1)
    groups = combined.map(normalize_group_text)
    y = df["fraudulent"].to_numpy()
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    trainval, test = next(outer.split(df, y, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE + 1)
    tr_rel, va_rel = next(inner.split(df.iloc[trainval], y[trainval], groups.iloc[trainval]))
    return trainval[tr_rel], trainval[va_rel], test


def metric_dict(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float | int | list]:
    pred = (score >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "confusion_matrix": cm.tolist(),
    }


def choose_threshold(y: np.ndarray, score: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, score)
    f2 = 5 * precision[:-1] * recall[:-1] / (4 * precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(np.nanargmax(f2))])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/fake_job_postings.csv")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    outdir = Path(args.output)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.data)
    data = engineer(raw)
    y = raw["fraudulent"].to_numpy()
    train_idx, val_idx, test_idx = split_groups(raw)
    split = {"train": train_idx.tolist(), "validation": val_idx.tolist(), "test": test_idx.tolist()}
    (outdir / "split_indices.json").write_text(json.dumps(split), encoding="utf-8")

    numeric_columns = BINARY_COLUMNS + [
        "salary_low", "salary_high", "salary_width", "salary_missing"
    ] + [c for c in data.columns if any(c.endswith(s) for s in [
        "_chars", "_words", "_missing", "_url_count", "_email_count",
        "_digit_ratio", "_upper_ratio", "_bang_count",
    ]) and c != "salary_missing"]
    structured = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=3), CATEGORICAL_COLUMNS),
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler(with_mean=False)),
        ]), numeric_columns),
    ])
    text = build_text(data)
    word = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=2,
        max_df=0.995, max_features=120_000, sublinear_tf=True, dtype=np.float32,
    )
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=80_000,
        sublinear_tf=True, dtype=np.float32,
    )

    Xs_train = structured.fit_transform(data.iloc[train_idx])
    Xs_val = structured.transform(data.iloc[val_idx])
    Xs_test = structured.transform(data.iloc[test_idx])
    Xw_train = word.fit_transform(text.iloc[train_idx])
    Xw_val = word.transform(text.iloc[val_idx])
    Xw_test = word.transform(text.iloc[test_idx])
    Xc_train = char.fit_transform(text.iloc[train_idx])
    Xc_val = char.transform(text.iloc[val_idx])
    Xc_test = char.transform(text.iloc[test_idx])

    matrices = {
        "structured": (Xs_train, Xs_val, Xs_test),
        "word_text": (Xw_train, Xw_val, Xw_test),
        "fused": (
            hstack([Xw_train, Xc_train, Xs_train], format="csr"),
            hstack([Xw_val, Xc_val, Xs_val], format="csr"),
            hstack([Xw_test, Xc_test, Xs_test], format="csr"),
        ),
    }
    results = {}
    models = {}
    for name, (xtr, xva, xte) in matrices.items():
        model = LogisticRegression(
            C=2.0 if name == "fused" else 1.0, class_weight="balanced",
            max_iter=1500, solver="liblinear", random_state=RANDOM_STATE,
        )
        model.fit(xtr, y[train_idx])
        val_score = model.predict_proba(xva)[:, 1]
        test_score = model.predict_proba(xte)[:, 1]
        threshold = choose_threshold(y[val_idx], val_score)
        results[name] = {
            "validation_selected_threshold": threshold,
            "validation": metric_dict(y[val_idx], val_score, threshold),
            "test_at_0_5": metric_dict(y[test_idx], test_score, 0.5),
            "test_selected": metric_dict(y[test_idx], test_score, threshold),
            "n_features": int(xtr.shape[1]),
        }
        models[name] = model
        if name == "fused":
            np.save(outdir / "test_scores.npy", test_score)

    # Bootstrap confidence intervals for the final model's test PR AUC and ROC AUC.
    rng = np.random.default_rng(RANDOM_STATE)
    final_score = np.load(outdir / "test_scores.npy")
    yt = y[test_idx]
    boot = {"pr_auc": [], "roc_auc": []}
    for _ in range(1000):
        idx = rng.integers(0, len(yt), len(yt))
        if np.unique(yt[idx]).size < 2:
            continue
        boot["pr_auc"].append(average_precision_score(yt[idx], final_score[idx]))
        boot["roc_auc"].append(roc_auc_score(yt[idx], final_score[idx]))
    results["fused"]["bootstrap_95_ci"] = {
        k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot.items()
    }

    # Feature interpretation.
    feature_names = np.concatenate([
        np.array(["word:" + x for x in word.get_feature_names_out()]),
        np.array(["char:" + x for x in char.get_feature_names_out()]),
        np.array(["structured:" + x for x in structured.get_feature_names_out()]),
    ])
    coef = models["fused"].coef_[0]
    order_pos = np.argsort(coef)[-30:][::-1]
    order_neg = np.argsort(coef)[:30]
    importance = pd.concat([
        pd.DataFrame({"direction": "fraud", "feature": feature_names[order_pos], "coefficient": coef[order_pos]}),
        pd.DataFrame({"direction": "legitimate", "feature": feature_names[order_neg], "coefficient": coef[order_neg]}),
    ])
    importance.to_csv(outdir / "feature_importance.csv", index=False)

    threshold = results["fused"]["validation_selected_threshold"]
    pred = (final_score >= threshold).astype(int)
    errors = raw.iloc[test_idx].copy()
    errors["score"] = final_score
    errors["prediction"] = pred
    errors["error_type"] = np.where((yt == 1) & (pred == 0), "false_negative",
                             np.where((yt == 0) & (pred == 1), "false_positive", "correct"))
    errors[errors.error_type != "correct"].sort_values("score", ascending=False).to_csv(
        outdir / "test_errors.csv", index=False
    )

    metadata = {
        "rows": len(raw), "positive_rows": int(y.sum()), "positive_rate": float(y.mean()),
        "unique_text_groups": int(raw[TEXT_COLUMNS].fillna("").astype(str).agg(" ".join, axis=1).map(normalize_group_text).nunique()),
        "split_sizes": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
        "split_positive_rates": {"train": float(y[train_idx].mean()), "validation": float(y[val_idx].mean()), "test": float(y[test_idx].mean())},
        "missing": raw.isna().sum().to_dict(),
    }
    (outdir / "metrics.json").write_text(json.dumps({"metadata": metadata, "models": results}, indent=2), encoding="utf-8")

    artifact = {
        "structured_transformer": structured, "word_vectorizer": word, "char_vectorizer": char,
        "model": models["fused"], "threshold": threshold, "text_columns": TEXT_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS, "numeric_columns": numeric_columns,
    }
    joblib.dump(artifact, outdir / "fake_job_detector.joblib", compress=3)

    # Report figures.
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    missing = raw.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    ax.barh(missing.index[::-1], (missing.values[::-1] * 100), color="#2E5B88")
    ax.set_xlabel("Missing values (%)"); ax.set_ylabel(""); ax.set_title("Missingness by field")
    fig.tight_layout(); fig.savefig(outdir / "figures" / "missingness.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    names = list(matrices)
    vals = [results[n]["test_selected"]["pr_auc"] for n in names]
    ax.bar(names, vals, color=["#9AA7B2", "#608FB5", "#183B5B"])
    ax.set_ylim(0, 1); ax.set_ylabel("Test PR AUC"); ax.set_title("Model comparison on grouped holdout")
    for i, v in enumerate(vals): ax.text(i, v + 0.025, f"{v:.3f}", ha="center")
    fig.tight_layout(); fig.savefig(outdir / "figures" / "model_comparison.png", dpi=180); plt.close(fig)

    precision, recall, _ = precision_recall_curve(yt, final_score)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(recall, precision, color="#183B5B", lw=2)
    ax.axhline(y.mean(), color="#999999", ls="--", label=f"Dataset prevalence {y.mean():.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Fused model precision recall curve")
    ax.legend(loc="lower left"); fig.tight_layout()
    fig.savefig(outdir / "figures" / "precision_recall.png", dpi=180); plt.close(fig)

    print(json.dumps({"metadata": metadata, "models": results}, indent=2))


if __name__ == "__main__":
    main()
