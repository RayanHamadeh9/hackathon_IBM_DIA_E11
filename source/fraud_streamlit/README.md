# Fraud Intelligence • Streamlit Control Room

A multi-page Streamlit app to explore the IBM fraud hackathon datasets, visualize scores, and run inference with your trained model.

## 1) Setup

```bash
cd fraud_streamlit
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

## 2) Place the app inside your project

Copy the entire `fraud_streamlit/` folder into your project root (the same level as the `hackthon data/` and `outputs/` folders).  
If your files live elsewhere, set the "Project base directory" in the sidebar to point at that folder.

Expected structure:
```text
<project-root>/
  hackthon data/
  outputs/
  dashboard/ (optional: daily_metrics.parquet, mcc_metrics.parquet)
  fraud_streamlit/
    app.py
    pages/
    utils.py
    requirements.txt
    .streamlit/config.toml
```

## 3) Run

```bash
streamlit run fraud_streamlit/app.py
```

Open the browser URL it prints.

## 4) Notes

- The app autodetects column names for dates, amounts, MCC, IDs, etc. It works out-of-the-box with the hackathon artifacts you produced (`eval_scored.parquet`, `train_features.parquet`, `submission.csv`, etc.).
- If `outputs/model_meta.joblib` contains a key like `best_thr` the app will use it as the default decision threshold. You can override it from the sidebar.
- Inference tab uses `outputs/model_final.joblib`. Upload a CSV with the same schema as your evaluation features.
- The Monitoring tab uses `dashboard/daily_metrics.parquet` when available; otherwise it's computed on the fly from the evaluation features and scores.
