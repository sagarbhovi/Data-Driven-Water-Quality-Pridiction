"""
Water Quality Monitoring System - Streamlit Dashboard
Final Year Project
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px

from config import (
    DATA_DIR,
    TRAIN_DIR,
    TEST_DIR,
    QUALITY_LABELS,
    QUALITY_COLORS
)

from predict_utils import load_model, predict_quality, FEATURES


# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Water Quality Monitoring System",
    page_icon="💧",
    layout="wide",
)

# ---------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------

def load_all_data():
    """Load train + test + uploaded data."""
    dfs = []

    train_path = os.path.join(TRAIN_DIR, "train.csv")
    test_path = os.path.join(TEST_DIR, "test.csv")

    if os.path.exists(train_path):
        dfs.append(pd.read_csv(train_path))

    if os.path.exists(test_path):
        dfs.append(pd.read_csv(test_path))

    # Include uploaded data
    if "uploaded_data" in st.session_state:
        dfs.append(st.session_state["uploaded_data"])

    if len(dfs) == 0:
        return None

    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

def render_sidebar():

    with st.sidebar:

        st.title("💧 Water Monitor")

        page = st.radio(
            "Navigation",
            [
                "About Project",
                "Dashboard",
                "Input & Predict",
                "City-wise Quality",
                "Upload Data",
            ],
        )

        st.markdown("---")

        if st.button("🔄 Refresh Dashboard"):
            st.rerun()

        st.caption("Final Year Project")

    return page


# ---------------------------------------------------
# ABOUT PAGE
# ---------------------------------------------------

def page_about():

    st.title("Water Quality Monitoring & prediction System")

    st.markdown(
        """
This system predicts **water quality levels** using machine learning
based on multiple physical and chemical parameters.

### Objectives
- Predict water quality
- Monitor city-wise water conditions
- Provide improvement suggestions
- Enable CSV data upload for analysis

### Technologies
- Python
- Streamlit
- Scikit-learn
- Pandas
- Plotly
"""
    )

    df = load_all_data()

    if df is not None:

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Samples", len(df))
        col2.metric("Cities", df["city"].nunique())
        col3.metric("Good Quality %",
                    round((df["quality"] == "Good").mean()*100, 2))


# ---------------------------------------------------
# DASHBOARD PAGE
# ---------------------------------------------------

def page_dashboard(df):

    st.title("📊 Water Quality Dashboard")

    if df is None or df.empty:
        st.warning("No data available.")
        return

    total = len(df)
    cities = df["city"].nunique()

    st.write(f"Total Samples: **{total}** | Cities: **{cities}**")

    col1, col2, col3, col4 = st.columns(4)

    for col, label in zip([col1, col2, col3, col4], QUALITY_LABELS):

        count = (df["quality"] == label).sum()
        col.metric(label, count)

    st.divider()

    # PIE CHART

    st.subheader("Quality Distribution")

    fig = px.pie(
        df,
        names="quality",
        color="quality",
        color_discrete_map=QUALITY_COLORS
    )

    st.plotly_chart(fig, use_container_width=True)

    # PARAMETER AVERAGES

    st.subheader("Average Parameters by Quality")

    numeric_cols = [
        "ph",
        "turbidity",
        "dissolved_oxygen",
        "total_dissolved_solids",
        "nitrate",
    ]

    avg = df.groupby("quality")[numeric_cols].mean().round(2)

    st.dataframe(avg, use_container_width=True)

    # RECENT DATA

    st.subheader("Recent Samples")

    st.dataframe(df.tail(50), use_container_width=True)


# ---------------------------------------------------
# INPUT PREDICTION PAGE
# ---------------------------------------------------

def page_predict():

    st.title("🔬 Predict Water Quality")

    with st.form("predict_form"):

        col1, col2 = st.columns(2)

        with col1:

            ph = st.slider("pH", 4.0, 10.0, 7.0)
            turbidity = st.number_input("Turbidity", 0.0, 100.0, 5.0)
            dissolved_oxygen = st.number_input("Dissolved Oxygen", 0.0, 20.0, 8.0)
            temperature = st.number_input("Temperature", 0.0, 40.0, 25.0)

        with col2:

            conductivity = st.number_input("Conductivity", 0.0, 3000.0, 500.0)
            tds = st.number_input("TDS", 0.0, 2500.0, 300.0)
            nitrate = st.number_input("Nitrate", 0.0, 50.0, 5.0)
            fecal_coliform = st.number_input("Fecal Coliform", 0, 500, 0)
            bod = st.number_input("BOD", 0.0, 30.0, 2.0)

        submit = st.form_submit_button("Predict")

    if submit:

        params = {
            "ph": ph,
            "turbidity": turbidity,
            "dissolved_oxygen": dissolved_oxygen,
            "temperature": temperature,
            "conductivity": conductivity,
            "total_dissolved_solids": tds,
            "nitrate": nitrate,
            "fecal_coliform": fecal_coliform,
            "bod": bod,
        }

        label, suggestions = predict_quality(params)

        if label:

            st.success(f"Predicted Quality: **{label}**")

            st.subheader("Suggestions")

            for s in suggestions:
                st.write("•", s)

        else:
            st.error("Model not loaded. Train model first.")


# ---------------------------------------------------
# CITY PAGE
# ---------------------------------------------------

def page_city(df):

    st.title("🏙️ City-wise Water Quality")

    if df is None:
        st.warning("No data available.")
        return

    city_quality = df.groupby("city")["quality"].value_counts().unstack(fill_value=0)

    city_quality = city_quality.reindex(columns=QUALITY_LABELS, fill_value=0)

    fig = px.bar(
        city_quality.reset_index().melt(
            id_vars="city",
            var_name="Quality",
            value_name="Count"
        ),
        x="city",
        y="Count",
        color="Quality",
        barmode="stack",
        color_discrete_map=QUALITY_COLORS
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dominant Quality per City")

    dominant = df.groupby("city")["quality"].agg(
        lambda x: x.mode().iloc[0]
    ).reset_index()

    dominant.columns = ["City", "Dominant Quality"]

    st.dataframe(dominant, use_container_width=True)


# ---------------------------------------------------
# UPLOAD PAGE
# ---------------------------------------------------

def page_upload():

    st.title("📁 Upload Water Data")

    st.info(
        "CSV must contain: city + water parameters"
    )

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:

        try:

            df = pd.read_csv(file)

            missing = [c for c in FEATURES if c not in df.columns]

            if missing:
                st.error(f"Missing columns: {missing}")
                return

            if "quality" not in df.columns:

                model, le = load_model()

                if model:

                    preds = model.predict(df[FEATURES])
                    df["quality"] = le.inverse_transform(preds)

            st.session_state["uploaded_data"] = df

            st.success(f"{len(df)} rows uploaded successfully")

            st.dataframe(df.head(20))

        except Exception as e:
            st.error(e)


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

def main():

    page = render_sidebar()

    df = load_all_data()

    if page == "About Project":
        page_about()

    elif page == "Dashboard":
        page_dashboard(df)

    elif page == "Input & Predict":
        page_predict()

    elif page == "City-wise Quality":
        page_city(df)

    elif page == "Upload Data":
        page_upload()


if __name__ == "__main__":
    main()