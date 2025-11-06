import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from utils import (
    get_paths, read_parquet_cached, read_csv_cached, read_json_cached,
    load_training_dataframe, load_eval_features, load_model, load_threshold,
    guess_datetime_col, guess_amount_col, guess_mcc_col, guess_id_col, guess_state_col, guess_card_type_col,
    attach_score_and_pred, compute_daily_metrics, compute_mcc_metrics
)

st.set_page_config(page_title="Fraud Intelligence • Control Room", page_icon="🛰️", layout="wide")

# ---------- Sidebar configuration ----------
st.sidebar.title("⚙️ Configuration")
base_dir = st.sidebar.text_input("Project base directory", value=str(Path(__file__).resolve().parent))
DATA_DIR, OUT_DIR, DASH_DIR = get_paths(base_dir)

st.sidebar.success(f"Data: {DATA_DIR.name} | Outputs: {OUT_DIR.name}")
threshold = load_threshold(base_dir)
st.sidebar.number_input("Decision threshold", min_value=0.0, max_value=1.0, value=float(threshold), step=0.01, key="ui_thr")

# ---------- Load data ----------
model = load_model(base_dir)
train_df = load_training_dataframe(base_dir)
eval_df = load_eval_features(base_dir)

# Attach scores/preds to evaluation if not already
eval_df = attach_score_and_pred(eval_df.copy(), st.session_state["ui_thr"])

# ---------- Hero / Overview ----------
st.title("🛰️ Fraud Intelligence Control Room")
st.caption("Hackathon IBM DIA • Detecting fraudulent card & online payments")

# KPIs
kpi_cols = st.columns(4)
# Training fraud rate (if label exists)
fraud_col = "fraud_label" if "fraud_label" in train_df.columns else ("label" if "label" in train_df.columns else None)
if fraud_col:
    train_fraud_rate = float(train_df[fraud_col].mean())
    kpi_cols[0].metric("Training fraud rate", f"{train_fraud_rate*100:.2f}%")
else:
    kpi_cols[0].metric("Training fraud rate", "—")

pred_rate = float(eval_df["pred"].mean()) if not eval_df.empty else 0.0
kpi_cols[1].metric("Evaluation predicted fraud rate", f"{pred_rate*100:.2f}%")
kpi_cols[2].metric("# Eval transactions", f"{len(eval_df):,}")
kpi_cols[3].metric("Decision threshold", f"{st.session_state['ui_thr']:.3f}")

st.divider()

# ---------- Tabs ----------
tab_overview, tab_scores, tab_segments, tab_monitor, tab_infer = st.tabs(
    ["Overview", "Score Distribution", "Segments & Categories", "Monitoring", "Run Inference"]
)

# ---- Overview tab ----
with tab_overview:
    date_col = guess_datetime_col(eval_df)
    amount_col = guess_amount_col(eval_df)
    id_col = guess_id_col(eval_df)
    mcc_col = guess_mcc_col(eval_df)

    left, right = st.columns((2,1), gap="large")
    with left:
        st.subheader("Daily transaction volume & predicted frauds")
        if date_col:
            daily = compute_daily_metrics(eval_df, date_col)
            if not daily.empty:
                base = alt.Chart(daily).encode(x="date:T")
                c1 = base.mark_bar().encode(y=alt.Y("txn_count:Q", title="Transaction Count"))
                c2 = base.mark_line(point=True).encode(y=alt.Y("pred_frauds:Q", title="Predicted Frauds"), tooltip=["date","pred_frauds","txn_count"])
                st.altair_chart(alt.layer(c1, c2).resolve_scale(y="independent").properties(height=260), use_container_width=True)
            else:
                st.info("No date column found to plot daily metrics.")
        else:
            st.info("No date column found to plot daily metrics.")

        st.subheader("Top 20 highest-risk transactions")
        if id_col:
            topk = eval_df.nlargest(20, "score")[[id_col, "score", "pred"]]
            st.dataframe(topk, hide_index=True, use_container_width=True)
        else:
            st.dataframe(eval_df.nlargest(20, "score")[["score","pred"]], hide_index=True, use_container_width=True)

    with right:
        st.subheader("Score summary")
        if not eval_df.empty:
            st.write(pd.DataFrame({
                "min": [eval_df["score"].min()],
                "mean": [eval_df["score"].mean()],
                "median": [eval_df["score"].median()],
                "max": [eval_df["score"].max()]
            }))
        #st.subheader("Card type distribution (predicted frauds)")
        # Try to derive card type if available
        #card_type_col = guess_card_type_col(eval_df)
        #if card_type_col:
        #    pie_df = eval_df[eval_df["pred"]==1][card_type_col].value_counts(normalize=True).rename("share").reset_index().rename(columns={"index":"card_type"})
        #    chart = alt.Chart(pie_df).mark_arc(innerRadius=40).encode(theta="share:Q", color="card_type:N", tooltip=["card_type","share"]).properties(height=250)
        #    st.altair_chart(chart, use_container_width=True)
        #else:
        #    st.info("No card type column found.")

# ---- Score distribution tab ----
with tab_scores:
    st.subheader("Score distribution on evaluation set")
    bins = st.slider("Bins", 10, 200, 60, 5)
    if not eval_df.empty:
        hist = alt.Chart(eval_df).mark_bar().encode(
            alt.X("score:Q", bin=alt.Bin(maxbins=bins), title="Fraud probability score"),
            y="count()",
            color=alt.value("#6B46C1")
        ).properties(height=280)
        thr_rule = alt.Chart(pd.DataFrame({"thr":[st.session_state["ui_thr"]]})).mark_rule(color="red").encode(x="thr:Q")
        st.altair_chart(hist + thr_rule, use_container_width=True)
    st.subheader("Score distribution by prediction class")
    if not eval_df.empty:
        st.altair_chart(
            alt.Chart(eval_df.assign(pred_label=np.where(eval_df["pred"]==1, "Pred Fraud", "Pred Non-Fraud")))
                .mark_bar()
                .encode(x=alt.X("score:Q", bin=alt.Bin(maxbins=60)), y="count()", color="pred_label:N"),
            use_container_width=True
        )

# ---- Segments & categories tab ----
with tab_segments:
    st.subheader("Transaction volume by category (MCC)")
    mcc_col = guess_mcc_col(eval_df)
    if mcc_col:
        mcc_metrics = compute_mcc_metrics(eval_df, mcc_col)
        st.dataframe(mcc_metrics.head(30), use_container_width=True, hide_index=True)
        st.altair_chart(
            alt.Chart(mcc_metrics.head(20)).mark_bar().encode(
                x="txn_count:Q", y=alt.Y(f"{mcc_col}:N", sort="-x"), tooltip=["txn_count","avg_score","pred_rate"]
            ).properties(height=400),
            use_container_width=True
        )
        st.subheader("Average fraud score by category")
        st.altair_chart(
            alt.Chart(mcc_metrics.head(20)).mark_bar().encode(
                x="avg_score:Q", y=alt.Y(f"{mcc_col}:N", sort="-x")
            ).properties(height=400),
            use_container_width=True
        )
    else:
        st.info("No MCC/category column found.")

    st.subheader("Fraud rate by hour / day-of-week")
    date_col = guess_datetime_col(eval_df)
    if date_col:
        d = eval_df.copy()
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
        d = d.dropna(subset=[date_col])
        d["hour"] = d[date_col].dt.hour
        d["dow"] = d[date_col].dt.day_name()
        h = d.groupby("hour")["pred"].mean().reset_index().rename(columns={"pred":"rate"})
        w = d.groupby("dow")["pred"].mean().reset_index().rename(columns={"pred":"rate"})
        st.altair_chart(alt.Chart(h).mark_bar().encode(x="hour:O", y="rate:Q"), use_container_width=True)
        st.altair_chart(alt.Chart(w).mark_bar().encode(x="dow:N", y="rate:Q"), use_container_width=True)
    else:
        st.info("No datetime column detected to compute time-based rates.")

# ---- Monitoring tab ----
with tab_monitor:
    st.subheader("Monitoring: daily metrics")
    # Use pre-computed dashboard parquet if available; otherwise compute
    daily_metrics_path = Path(DATA_DIR) / "dashboard" / "daily_metrics.parquet"
    if daily_metrics_path.exists():
        daily = read_parquet_cached(daily_metrics_path)
        date_col = guess_datetime_col(daily) or "date"
        if date_col != "date":
            daily = daily.rename(columns={date_col:"date"})
    else:
        date_col = guess_datetime_col(eval_df)
        daily = compute_daily_metrics(eval_df, date_col)

    if daily is not None and not daily.empty:
        base = alt.Chart(daily).encode(x="date:T")
        c1 = base.mark_bar().encode(y=alt.Y("txn_count:Q", title="Transaction Count"))
        c2 = base.mark_line(point=True).encode(y=alt.Y("avg_score:Q", title="Average Score"))
        c3 = base.mark_line(color="red").encode(y=alt.Y("pred_rate:Q", title="Predicted Fraud Rate"))
        st.altair_chart(alt.layer(c1, c2, c3).resolve_scale(y="independent").properties(height=320), use_container_width=True)
        st.download_button("Download daily metrics (CSV)", daily.to_csv(index=False).encode("utf-8"), file_name="daily_metrics.csv")
    else:
        st.info("No daily metrics available.")

# ---- Inference tab ----
with tab_infer:
    st.subheader("Scoring / Inference")
    if model is None:
        st.warning("No trained model found at outputs/model_final.joblib. Upload a model or retrain.")
    else:
        file = st.file_uploader("Upload a CSV of transactions to score (schema similar to evaluation_features.csv)", type=["csv"])
        if file:
            new_df = pd.read_csv(file)
            try:
                proba = (model.predict_proba(new_df)[:,1]
                         if hasattr(model, "predict_proba")
                         else model.decision_function(new_df))
                res = pd.DataFrame({
                    "transaction_id": new_df[guess_id_col(new_df) or "transaction_id"],
                    "fraud_probability": proba,
                    "fraud_prediction": (proba >= st.session_state["ui_thr"]).astype(int)
                })
                st.success(f"Scored {len(res):,} rows.")
                st.dataframe(res.head(50), hide_index=True, use_container_width=True)
                st.download_button("⬇️ Download predictions CSV", res.to_csv(index=False).encode("utf-8"), file_name="predictions.csv")
            except Exception as e:
                st.error(f"Could not run inference with the provided file: {e}")
