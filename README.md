# Fake Job Detection

This project trains a leakage-aware fraud detector for the supplied fake job postings dataset. It combines word and character TF-IDF features with categorical, binary, missingness, location, salary-format, and text-shape features.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python code/train.py --data data/fake_job_postings.csv --output outputs
PYTHONPATH=code python code/train_advanced.py
```

The training command creates evaluation metrics, figures, error cases, a saved model, and fixed split indices under `outputs/`.

## Score new data

```bash
PYTHONPATH=code python code/predict_advanced.py --input new_jobs.csv --output predictions.csv
```

The final model is a validation-weighted two-tower ensemble: XGBoost learns nonlinear interactions among structured, missingness, and text-shape features, while a word-plus-character TF-IDF tower models high-dimensional job text. The input must use the same columns as the training data except that `fraudulent` is optional. The output adds a risk score and a binary decision based on the validation-selected threshold.
