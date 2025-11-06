from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------- Paths ----------
@st.cache_resource
def get_paths(base_dir: Optional[str] = None) -> Tuple[Path, Path, Path]:
    """Return DATA_DIR, OUT_DIR, DASH_DIR; tolerate the 'hackthon data' typo in folder name."""
    base = Path(base_dir) if base_dir else Path(__file__).resolve().parent
    candidates = ["hackthon data", "hackathon data", "data"]
    data_dir = None
    for c in candidates:
        p = base / c
        if p.exists():
            data_dir = p
            break
    if data_dir is None:
        data_dir = base / "hackthon data"  # default
    out_dir = base / "outputs"
    dash_dir = base / "dashboard"
    return data_dir, out_dir, dash_dir

# ---------- Safe readers with caching ----------
@st.cache_data(show_spinner=False)
def read_parquet_cached(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(p)
    except Exception:
        # Fallback to pandas engine if pyarrow not available or file old-format
        return pd.read_parquet(p, engine="auto")

@st.cache_data(show_spinner=False)
def read_csv_cached(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)

@st.cache_data(show_spinner=False)
def read_json_cached(path: str | Path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r") as f:
        return json.load(f)

# ---------- Column guessers (schema-agnostic) ----------
def col_first(df: pd.DataFrame, options) -> Optional[str]:
    for c in options:
        if c in df.columns:
            return c
    return None

def guess_datetime_col(df: pd.DataFrame) -> Optional[str]:
    return col_first(df, ["datetime","tx_datetime","transaction_date","date","trans_date","trans_date_time","event_time","timestamp","Timestamp"])

def guess_amount_col(df: pd.DataFrame) -> Optional[str]:
    return col_first(df, ["amount","amt","transaction_amount","purchase_amount","trans_amount"])

def guess_id_col(df: pd.DataFrame) -> Optional[str]:
    return col_first(df, ["transaction_id","txn_id","id","Id"])

def guess_mcc_col(df: pd.DataFrame) -> Optional[str]:
    return col_first(df, ["mcc","mcc_code","merchant_category_code","merchant_mcc"])

def guess_state_col(df: pd.DataFrame) -> Optional[str]:
    return col_first(df, ["state","us_state","client_state","addr_state"])

def guess_card_type_col(df: pd.DataFrame) -> Optional[str]:
    return col_first(df, ["card_type","type","card_brand"])

# ---------- Core loaders ----------
@st.cache_data(show_spinner=False)
def load_training_dataframe(base_dir: Optional[str] = None) -> pd.DataFrame:
    data_dir, out_dir, _ = get_paths(base_dir)
    # Prefer engineered features if present
    feats = out_dir / "train_features.parquet"
    if feats.exists():
        df = read_parquet_cached(feats)
    else:
        df = read_csv_cached(data_dir / "transactions_train.csv")
        labels = read_json_cached(data_dir / "train_fraud_labels.json")
        if isinstance(labels, dict):
            lab = pd.Series(labels, name="fraud_label").reset_index().rename(columns={"index":"transaction_id"})
        elif isinstance(labels, list):
            lab = pd.DataFrame(labels)
            if "transaction_id" not in lab.columns or "fraud_label" not in lab.columns:
                lab = pd.DataFrame()  # unknown schema
        else:
            lab = pd.DataFrame()
        if not df.empty and not lab.empty:
            idc = guess_id_col(df) or "transaction_id"
            df = df.merge(lab, left_on=idc, right_on="transaction_id", how="left")
    return df

@st.cache_data(show_spinner=False)
def load_eval_features(base_dir: Optional[str] = None) -> pd.DataFrame:
    data_dir, out_dir, _ = get_paths(base_dir)
    # prefer parquet if user exported engineered features
    eval_scored = read_parquet_cached(out_dir / "eval_scored.parquet")
    if not eval_scored.empty:
        return eval_scored
    # otherwise raw evaluation features
    ev = read_csv_cached(data_dir / "evaluation_features.csv")
    # merge submission if exists
    sub = read_csv_cached(out_dir / "submission.csv")
    if not ev.empty and not sub.empty:
        idc = guess_id_col(ev) or "transaction_id"
        ev = ev.merge(sub, left_on=idc, right_on="transaction_id", how="left", suffixes=("","_sub"))
    return ev

@st.cache_resource
def load_model(base_dir: Optional[str] = None):
    _, out_dir, _ = get_paths(base_dir)
    model_path = out_dir / "model_final.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    return None

@st.cache_data(show_spinner=False)
def load_threshold(base_dir: Optional[str] = None) -> float:
    _, out_dir, _ = get_paths(base_dir)
    thr = 0.5
    meta_path = out_dir / "model_meta.joblib"
    if meta_path.exists():
        try:
            meta = joblib.load(meta_path)
            if isinstance(meta, dict):
                for k in ["best_thr","threshold","decision_threshold"]:
                    if k in meta:
                        thr = float(meta[k]); break
            else:
                d = getattr(meta, "__dict__", {})
                for k in ["best_thr","threshold","decision_threshold"]:
                    if k in d:
                        thr = float(d[k]); break
        except Exception:
            pass
    return float(thr)

# ---------- Metrics helpers ----------
def attach_score_and_pred(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Ensure df has 'score' and 'pred' columns derived from any available score/prediction fields."""
    sc_col = None
    for c in ["score","fraud_probability","proba","pred_score","prediction_score"]:
        if c in df.columns:
            sc_col = c; break
    if sc_col is None and "fraud_prediction" in df.columns:
        # Binary prediction only → cast to float
        df["score"] = df["fraud_prediction"].astype(float)
    elif sc_col is not None:
        df = df.rename(columns={sc_col: "score"})
    else:
        # No score available; default to zeros
        df["score"] = 0.0

    df["pred"] = (df["score"] >= threshold).astype(int)
    return df

def compute_daily_metrics(df: pd.DataFrame, date_col: Optional[str]) -> pd.DataFrame:
    if date_col is None:
        return pd.DataFrame()
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])
    g = (
        d.groupby(pd.Grouper(key=date_col, freq="D"))
         .agg(txn_count=("pred","size"),
              avg_score=("score","mean"),
              pred_frauds=("pred","sum"))
         .reset_index()
         .rename(columns={date_col:"date"})
    )
    g["pred_rate"] = np.where(g["txn_count"]>0, g["pred_frauds"]/g["txn_count"], 0.0)
    return g

def compute_mcc_metrics(df: pd.DataFrame, mcc_col: Optional[str]) -> pd.DataFrame:
    if mcc_col is None:
        return pd.DataFrame()
    g = (
        df.groupby(mcc_col)
          .agg(txn_count=("pred","size"),
               avg_score=("score","mean"),
               pred_frauds=("pred","sum"))
          .reset_index()
          .sort_values("avg_score", ascending=False)
    )
    g["pred_rate"] = np.where(g["txn_count"]>0, g["pred_frauds"]/g["txn_count"], 0.0)
    return g

def pick_column(df: pd.DataFrame, label: str, candidates: list[str]) -> Optional[str]:
    c = [c for c in candidates if c in df.columns]
    return c[0] if c else None
