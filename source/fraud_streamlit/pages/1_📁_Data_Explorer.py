from pathlib import Path
import pandas as pd
import streamlit as st
from utils import get_paths, read_parquet_cached, read_csv_cached

st.set_page_config(page_title="Data Explorer", page_icon="📁", layout="wide")

st.title("📁 Data Explorer")

base_dir = st.sidebar.text_input("Project base directory", value=str(Path(__file__).resolve().parent.parent))
DATA_DIR, OUT_DIR, DASH_DIR = get_paths(base_dir)

options = {
    "transactions_train.csv": DATA_DIR / "transactions_train.csv",
    "users_data.csv": DATA_DIR / "users_data.csv",
    "cards_data.csv": DATA_DIR / "cards_data.csv",
    "evaluation_features.csv": DATA_DIR / "evaluation_features.csv",
    "train_features.parquet (outputs)": OUT_DIR / "train_features.parquet",
    "gold_features.parquet (outputs)": OUT_DIR / "gold_features.parquet",
    "eval_scored.parquet (outputs)": OUT_DIR / "eval_scored.parquet",
    "submission.csv (outputs)": OUT_DIR / "submission.csv",
    "daily_metrics.parquet (dashboard)": (Path(base_dir) / "dashboard" / "daily_metrics.parquet"),
    "mcc_metrics.parquet (dashboard)": (Path(base_dir) / "dashboard" / "mcc_metrics.parquet"),
}

choice = st.selectbox("Choose a dataset to preview", list(options.keys()))
path = options[choice]

if str(path).endswith(".parquet"):
    df = read_parquet_cached(path)
elif str(path).endswith(".csv"):
    df = read_csv_cached(path)
else:
    df = pd.DataFrame()

if df is None or df.empty:
    st.warning("File not found or empty.")
else:
    st.write(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} cols")
    st.dataframe(df.head(200), use_container_width=True)
    st.download_button("⬇️ Download sample (first 5k rows)",
                       df.head(5000).to_csv(index=False).encode("utf-8"),
                       file_name=Path(choice).name.replace(".parquet",".csv"))
