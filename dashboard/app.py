import pickle
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

try:
    import shap
    import matplotlib.pyplot as plt
except Exception:
    shap = None
    plt = None


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_PATH = APP_DIR / "model.pkl"
SCORED_PATH = APP_DIR / "scored_transactions.csv"

st.set_page_config(page_title="Fraud Dashboard", layout="wide")


@st.cache_data
def load_scored_transactions():
    if SCORED_PATH.exists():
        return pd.read_csv(SCORED_PATH)
    return pd.DataFrame()


@st.cache_resource
def load_model_artifact():
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def add_engineered_features(data):
    data = data.copy()
    amount_mean = data["TransactionAmt"].mean()
    amount_std = data["TransactionAmt"].std()
    data["AmtToMeanRatio"] = data["TransactionAmt"] / max(amount_mean, 1e-9)
    data["AmtZScore"] = (data["TransactionAmt"] - amount_mean) / max(amount_std, 1e-9)
    data["HourOfDay"] = ((data["TransactionDT"] // 3600) % 24).astype(int)
    data["DayOfWeek"] = ((data["TransactionDT"] // (3600 * 24)) % 7).astype(int)
    device_type = data.get("DeviceType", pd.Series("unknown", index=data.index)).fillna("unknown").astype(str).str.lower()
    device_info = data.get("DeviceInfo", pd.Series("", index=data.index)).fillna("").astype(str).str.lower()
    data["DeviceRisk"] = (
        device_type.eq("mobile")
        | device_info.str.contains("android|ios|samsung|huawei|moto|lg|iphone", regex=True)
    ).astype(int)
    return data


def load_raw_data():
    transaction_path = DATA_DIR / "train_transaction.csv"
    identity_path = DATA_DIR / "train_identity.csv"
    if not transaction_path.exists() or not identity_path.exists():
        return pd.DataFrame()
    transaction = pd.read_csv(transaction_path)
    identity = pd.read_csv(identity_path)
    return transaction.merge(identity, on="TransactionID", how="left")


st.title("Fraud Detection Dashboard")

scored = load_scored_transactions()
artifact = load_model_artifact()

with st.sidebar:
    st.header("Filters")
    page = st.radio("Page", ["Overview", "Transaction Explorer", "SHAP Explainer"])
    if not scored.empty and "risk_tier" in scored.columns:
        selected_tiers = st.multiselect(
            "Risk Tier",
            options=sorted(scored["risk_tier"].dropna().unique()),
            default=sorted(scored["risk_tier"].dropna().unique()),
        )
    else:
        selected_tiers = []

if scored.empty:
    st.info(
        "Run analysis.ipynb first. It trains the model and creates dashboard/model.pkl "
        "plus dashboard/scored_transactions.csv."
    )
    st.stop()

if selected_tiers and "risk_tier" in scored.columns:
    visible = scored[scored["risk_tier"].isin(selected_tiers)].copy()
else:
    visible = scored.copy()

if page == "Overview":
    total_transactions = len(visible)
    fraud_count = int(visible["isFraud"].sum()) if "isFraud" in visible.columns else 0
    detection_rate = float((visible.get("fraud_probability", pd.Series(dtype=float)) >= 0.40).mean()) if total_transactions else 0
    avg_fraud_amount = visible.loc[visible.get("isFraud", 0).eq(1), "TransactionAmt"].mean() if "TransactionAmt" in visible.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{total_transactions:,}")
    c2.metric("Known Fraud Count", f"{fraud_count:,}")
    c3.metric("Detection Rate", f"{detection_rate:.2%}")
    c4.metric("Average Fraud Amount", f"${avg_fraud_amount:,.2f}" if pd.notna(avg_fraud_amount) else "$0.00")

    left, right = st.columns(2)
    with left:
        tier_counts = visible["risk_tier"].value_counts().reset_index()
        tier_counts.columns = ["risk_tier", "count"]
        st.plotly_chart(px.pie(tier_counts, names="risk_tier", values="count", hole=0.45, title="Risk Tier Mix"), use_container_width=True)
    with right:
        if "HourOfDay" in visible.columns:
            hourly = visible.groupby("HourOfDay")["isFraud"].mean().reset_index()
            st.plotly_chart(px.line(hourly, x="HourOfDay", y="isFraud", markers=True, title="Fraud Rate by Hour"), use_container_width=True)

elif page == "Transaction Explorer":
    search_id = st.text_input("Search by TransactionID")
    min_prob, max_prob = st.slider("Fraud Probability Range", 0.0, 1.0, (0.0, 1.0), 0.01)
    table = visible[
        visible["fraud_probability"].between(min_prob, max_prob)
    ].copy()
    if search_id:
        table = table[table["TransactionID"].astype(str).str.contains(search_id.strip(), case=False, na=False)]

    st.dataframe(table.sort_values("fraud_probability", ascending=False), use_container_width=True, height=520)

    if not table.empty:
        st.plotly_chart(
            px.scatter(
                table.head(5000),
                x="HourOfDay",
                y="TransactionAmt",
                color="fraud_probability",
                hover_data=["TransactionID", "risk_tier", "isFraud"],
                color_continuous_scale="Reds",
                title="Transaction Amount vs Hour by Fraud Probability",
            ),
            use_container_width=True,
        )

elif page == "SHAP Explainer":
    st.subheader("Transaction-Level Explanation")
    transaction_id = st.text_input("Enter TransactionID")

    if not transaction_id:
        st.caption("Enter a TransactionID from the explorer to generate a local explanation.")
        st.stop()

    selected = visible[visible["TransactionID"].astype(str).eq(transaction_id.strip())]
    if selected.empty:
        st.warning("TransactionID was not found in scored transactions.")
        st.stop()

    row = selected.iloc[0]
    st.metric("Fraud Probability", f"{row['fraud_probability']:.2%}")
    st.metric("Risk Tier", row["risk_tier"])
    st.write(
        "This score is based on the transaction amount, time, device information, "
        "and other encoded transaction features."
    )

    if artifact is None or shap is None:
        st.info("SHAP plot requires dashboard/model.pkl and the shap package.")
        st.stop()

    raw = load_raw_data()
    if raw.empty:
        st.info("Place Kaggle CSV files in data/ to compute live SHAP explanations.")
        st.stop()

    raw_selected = raw[raw["TransactionID"].astype(str).eq(transaction_id.strip())]
    if raw_selected.empty:
        st.warning("Raw transaction was not found in data files.")
        st.stop()

    model = artifact["model"]
    preprocessor = artifact["preprocessor"]
    drop_columns = artifact.get("drop_columns", [])
    feature_names = artifact["feature_names"]

    prepared = add_engineered_features(raw_selected.drop(columns=drop_columns, errors="ignore"))
    model_input = prepared.drop(columns=["isFraud", "TransactionID"], errors="ignore")
    transformed = preprocessor.transform(model_input)
    transformed_df = pd.DataFrame(transformed, columns=feature_names)

    try:
        explainer = shap.TreeExplainer(model)
        values = explainer(transformed_df)
        fig = plt.figure()
        shap.plots.waterfall(values[0], max_display=15, show=False)
        st.pyplot(fig, clear_figure=True)
    except Exception as exc:
        st.warning(f"Could not render SHAP waterfall for this model: {exc}")
