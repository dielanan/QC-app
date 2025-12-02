import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# -----------------------------------------------------
# 1. Load prediction library (relative import)
# -----------------------------------------------------
from be_qc_lib_saved import predict_new


# -----------------------------------------------------
# 2. Auto-detect paths (GitHub + Streamlit compatible)
# -----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "be_qc_models")
LOOKUP_DIR = os.path.join(BASE_DIR, "lookup")

# -----------------------------------------------------
# 3. Load lookup tables (CSV)
# -----------------------------------------------------
df_hierarchy = pd.read_csv(os.path.join(LOOKUP_DIR, "lookup_sektor_subsektor_msic.csv"))
df_nd = pd.read_csv(os.path.join(LOOKUP_DIR, "lookup_negeri_daerah.csv"))

# -----------------------------------------------------
# 4. Targets + Features
# -----------------------------------------------------
TARGETS = ["OUTPUT", "INPUT", "NILAI_DITAMBAH", "GAJI_UPAH", "JUMLAH_PEKERJA"]

FEATURES = {
    "OUTPUT": {"num": ["JUMLAH_PEKERJA", "HARTA_TETAP", "GAJI_UPAH", "OUTPUT"]},
    "INPUT": {"num": ["JUMLAH_PEKERJA", "HARTA_TETAP", "OUTPUT", "INPUT"]},
    "NILAI_DITAMBAH": {"num": ["OUTPUT", "INPUT", "JUMLAH_PEKERJA", "HARTA_TETAP", "NILAI_DITAMBAH"]},
    "GAJI_UPAH": {"num": ["JUMLAH_PEKERJA", "OUTPUT", "HARTA_TETAP", "GAJI_UPAH"]},
    "JUMLAH_PEKERJA": {"num": ["OUTPUT", "INPUT", "NILAI_DITAMBAH", "GAJI_UPAH", "HARTA_TETAP", "JUMLAH_PEKERJA"]}
}

# -----------------------------------------------------
# UI Header
# -----------------------------------------------------
st.title("BE ML-Driven QC")

mode = st.radio("Select Mode:", ["Single Input", "Batch (CSV Upload)"], horizontal=True)
selected = st.radio("Target Variable:", TARGETS, horizontal=True)


# =======================================================================
# 5. SINGLE INPUT MODE
# =======================================================================
if mode == "Single Input":

    st.sidebar.title(f"Input Data — {selected}")
    user_input = {}
    feats = FEATURES[selected]

    # SEKTOR
    sektor_list = sorted(df_hierarchy["SEKTOR"].unique())
    sektor = st.sidebar.selectbox("SEKTOR", sektor_list)
    user_input["SEKTOR"] = sektor

    # SUBSEKTOR
    sub_opts = sorted(df_hierarchy[df_hierarchy["SEKTOR"] == sektor]["SUBSEKTOR"].unique())
    subsektor = st.sidebar.selectbox("SUBSEKTOR", sub_opts)
    user_input["SUBSEKTOR"] = subsektor

    # MSIC
    msic_opts = sorted(df_hierarchy[
        (df_hierarchy["SEKTOR"] == sektor) &
        (df_hierarchy["SUBSEKTOR"] == subsektor)
    ]["MSIC_5D"].unique())
    msic = st.sidebar.selectbox("MSIC 5D", msic_opts)
    user_input["MSIC_5D"] = msic

    # NEGERI
    negeri = st.sidebar.selectbox("NEGERI", sorted(df_nd["NEGERI"].unique()))
    user_input["NEGERI"] = negeri

    # DAERAH
    daerah_opts = sorted(df_nd[df_nd["NEGERI"] == negeri]["DAERAH"].unique())
    daerah = st.sidebar.selectbox("DAERAH", daerah_opts)
    user_input["DAERAH"] = daerah

    # Numeric inputs
    for col in feats["num"]:
        if col == "JUMLAH_PEKERJA":
            user_input[col] = st.sidebar.number_input(col, min_value=0, step=1)
        else:
            user_input[col] = st.sidebar.number_input(col, min_value=0.0)

    run = st.sidebar.button(f"Run QC for {selected}")

    # ----------------------------------------------------------
    # Run prediction
    # ----------------------------------------------------------
    if run:
        df_input = pd.DataFrame([user_input])
        result = predict_new(df_input, out_dir=MODEL_DIR)

        st.subheader("Prediction Result")
        st.dataframe(result)

        # Extract values
        low = result.filter(like="_PRED_LOW").iloc[0][0]
        med = result.filter(like="_PRED_MED").iloc[0][0]
        up = result.filter(like="_PRED_UP").iloc[0][0]
        actual = user_input[selected]

        # Determine flag
        if actual < low:
            flag_color = "red"
            explanation = "🔴 UNDER-reporting (actual < lower bound)"
        elif actual > up:
            flag_color = "red"
            explanation = "🔴 OVER-reporting (actual > upper bound)"
        else:
            flag_color = "green"
            explanation = "🟢 Within range (no issue)"

        st.info(explanation)

        # Plot
        fig = go.Figure()
        fig.add_vrect(x0=low, x1=up, fillcolor="lightgreen", opacity=0.3)
        fig.add_vline(x=med, line_color="black")
        fig.add_trace(go.Scatter(
            x=[actual], y=[0],
            mode="markers+text",
            marker=dict(color=flag_color, size=14),
            text=[f"{actual:,.2f}"],
            textposition="top center"
        ))
        fig.update_layout(
            xaxis_title="Value",
            yaxis=dict(showticklabels=False),
            height=260
        )
        st.plotly_chart(fig, use_container_width=True)


# =======================================================================
# 6. BATCH MODE
# =======================================================================
if mode == "Batch (CSV Upload)":

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df_batch = pd.read_csv(uploaded_file)
        st.write("First 5 rows:")
        st.dataframe(df_batch.head())

        if st.button("Run Batch Prediction"):
            result = predict_new(df_batch, out_dir=MODEL_DIR)

            # flagging
            low = result.filter(like="_PRED_LOW")
            up = result.filter(like="_PRED_UP")
            med = result.filter(like="_PRED_MED")

            st.subheader("Batch Prediction Completed")
            st.dataframe(result)

            st.download_button(
                "Download Full Results",
                result.to_csv(index=False).encode("utf-8"),
                file_name="batch_qc_results.csv",
                mime="text/csv"
            )
