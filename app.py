import streamlit as st
import pandas as pd
import sys
import plotly.express as px
import plotly.graph_objects as go
import os

# -----------------------------------------------------
# FIXED: Auto-detect base directory (GitHub-friendly)
# -----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------
# Load prediction library (NO MORE D:\ PATH)
# -----------------------------------------------------
sys.path.append(BASE_DIR)
from be_qc_lib_saved import predict_new

# -----------------------------------------------------
# FIXED PATHS (relative — works on GitHub)
# -----------------------------------------------------
MODEL_DIR = os.path.join(BASE_DIR, "be_qc_models")
LOOKUP = os.path.join(BASE_DIR, "lookup")

# -----------------------------------------------------
# Load lookups (dependency)
# -----------------------------------------------------
df_hierarchy = pd.read_csv(os.path.join(LOOKUP, "lookup_sektor_subsektor_msic.csv"))
df_nd = pd.read_csv(os.path.join(LOOKUP, "lookup_negeri_daerah.csv"))

# -----------------------------------------------------
# Targets + features
# -----------------------------------------------------
TARGETS = ["OUTPUT", "INPUT", "NILAI_DITAMBAH", "GAJI_UPAH", "JUMLAH_PEKERJA"]

FEATURES = {
    "OUTPUT": {"num": ["JUMLAH_PEKERJA","HARTA_TETAP","GAJI_UPAH","OUTPUT"]},
    "INPUT": {"num": ["JUMLAH_PEKERJA","HARTA_TETAP","OUTPUT","INPUT"]},
    "NILAI_DITAMBAH": {"num": ["OUTPUT","INPUT","JUMLAH_PEKERJA","HARTA_TETAP","NILAI_DITAMBAH"]},
    "GAJI_UPAH": {"num": ["JUMLAH_PEKERJA","OUTPUT","HARTA_TETAP","GAJI_UPAH"]},
    "JUMLAH_PEKERJA": {"num": ["OUTPUT","INPUT","NILAI_DITAMBAH","GAJI_UPAH","HARTA_TETAP","JUMLAH_PEKERJA"]}
}

# -----------------------------------------------------
# UI Header
# -----------------------------------------------------
st.title("BE ML-Driven QC")

# -----------------------------------------------------
# MODE SELECTOR
# -----------------------------------------------------
mode = st.radio("Select Mode:", ["Single Input", "Batch (CSV Upload)"], horizontal=True)

selected = st.radio("Select Target:", TARGETS, index=0, horizontal=True)

# =======================================================================
# MODE 1 — SINGLE INPUT
# =======================================================================
if mode == "Single Input":
    
    st.sidebar.title(f"Input Data — {selected}")
    user_input = {}
    feats = FEATURES[selected]

    # -------------------------------
    # DEPENDENCY DROPDOWNS
    # -------------------------------
    sektor_list = sorted(df_hierarchy["SEKTOR"].unique())
    sektor = st.sidebar.selectbox("SEKTOR", sektor_list, key=f"{selected}_sektor")
    user_input["SEKTOR"] = sektor

    sub_opts = sorted(df_hierarchy[df_hierarchy["SEKTOR"] == sektor]["SUBSEKTOR"].unique())
    subsektor = st.sidebar.selectbox("SUBSEKTOR", sub_opts, key=f"{selected}_subsektor")
    user_input["SUBSEKTOR"] = subsektor

    msic_opts = sorted(df_hierarchy[
        (df_hierarchy["SEKTOR"] == sektor) &
        (df_hierarchy["SUBSEKTOR"] == subsektor)
    ]["MSIC_5D"].unique())
    msic = st.sidebar.selectbox("MSIC 5D", msic_opts, key=f"{selected}_msic")
    user_input["MSIC_5D"] = msic

    negeri_list = sorted(df_nd["NEGERI"].unique())
    negeri = st.sidebar.selectbox("NEGERI", negeri_list, key=f"{selected}_negeri")
    user_input["NEGERI"] = negeri

    daerah_opts = sorted(df_nd[df_nd["NEGERI"] == negeri]["DAERAH"].unique())
    daerah = st.sidebar.selectbox("DAERAH", daerah_opts, key=f"{selected}_daerah")
    user_input["DAERAH"] = daerah

    # numeric
    for col in feats["num"]:
        key = f"{selected}_num_{col}"
        if col == "JUMLAH_PEKERJA":
            user_input[col] = st.sidebar.number_input(col, min_value=0, step=1, key=key)
        else:
            user_input[col] = st.sidebar.number_input(col, min_value=0.0, format="%.2f", key=key)

    run = st.sidebar.button(f"Run QC for {selected}", key=f"run_{selected}")

    # =================================================================
    # RUN PREDICTION (Single Row)
    # =================================================================
    if run:
        df_input = pd.DataFrame([user_input])
        result = predict_new(df_input, out_dir=MODEL_DIR)

        st.subheader("Prediction Result")
        st.dataframe(result)

        # --- existing logic not changed ---
        low_col = next((c for c in result.columns if "low" in c.lower() and selected.lower() in c.lower()), None)
        med_col = next((c for c in result.columns if "med" in c.lower() and selected.lower() in c.lower()), None)
        up_col  = next((c for c in result.columns if "up"  in c.lower() and selected.lower() in c.lower()), None)

        if low_col and med_col and up_col:
            lb = float(result[low_col].iloc[0])
            mb = float(result[med_col].iloc[0])
            ub = float(result[up_col].iloc[0])
            actual = float(user_input.get(selected, 0))

            if actual < lb:
                flag_color = "red"
                explanation = "🔴 Below Lower Bound → Possible UNDER-reporting"
            elif actual > ub:
                flag_color = "red"
                explanation = "🔴 Above Upper Bound → Possible OVER-reporting"
            else:
                flag_color = "green"
                explanation = "🟢 Within Model Range → No anomaly"

            st.info(explanation)

            fig = go.Figure()
            fig.add_vrect(x0=lb, x1=ub, fillcolor="lightblue", opacity=0.3)
            fig.add_vline(x=lb, line_dash="dash", line_color="blue")
            fig.add_vline(x=mb, line_color="black")
            fig.add_vline(x=ub, line_dash="dash", line_color="blue")

            fig.add_trace(go.Scatter(
                x=[actual], y=[0],
                mode="markers+text",
                marker=dict(color=flag_color, size=14),
                text=[f"{actual:,.2f}"],
                textposition="top center"
            ))

            fig.update_layout(
                xaxis_title=f"{selected} Value",
                yaxis=dict(showticklabels=False),
                height=260
            )
            st.plotly_chart(fig, use_container_width=True)

        bar_df = pd.DataFrame({
            "Category": feats["num"],
            "Value": [user_input[v] for v in feats["num"]]
        })
        st.subheader("📊 Numeric Inputs Used")
        st.plotly_chart(px.bar(bar_df, x="Category", y="Value", text="Value"), use_container_width=True)


# =======================================================================
# MODE 2 — BATCH INPUT
# =======================================================================
if mode == "Batch (CSV Upload)":

    st.subheader("📁 Upload CSV file for batch QC prediction")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df_batch = pd.read_csv(uploaded_file)
        st.write("🔍 First 5 rows of input data:")
        st.dataframe(df_batch.head())

        if st.button("Run Batch Prediction"):

            result_batch = predict_new(df_batch, out_dir=MODEL_DIR)

            st.subheader("Batch Prediction Results")
            st.dataframe(result_batch)

            st.download_button(
                "Download Results",
                result_batch.to_csv(index=False).encode('utf-8'),
                file_name="batch_qc_results.csv",
                mime="text/csv"
            )
