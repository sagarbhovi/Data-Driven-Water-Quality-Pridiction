"""Water Quality Monitoring System - Streamlit Dashboard | """
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from config import (
    DATA_DIR, TRAIN_DIR, TEST_DIR, QUALITY_LABELS, QUALITY_COLORS,
    PARAM_RANGES,
)
from predict_utils import load_model, predict_quality, get_suggestions, FEATURES

st.set_page_config(page_title="Water Quality Monitoring System", page_icon="💧", layout="wide", initial_sidebar_state="expanded")

# Custom CSS - Final Year Project theme
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(165deg, #06121a 0%, #0c1929 25%, #0d2137 50%, #132f4c 75%, #0a1628 100%);
        font-family: 'Segoe UI', 'SF Pro Display', system-ui, sans-serif;
    }
    .main .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1400px; }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d2137 0%, #0a1628 100%);
        border-right: 1px solid rgba(0, 212, 255, 0.2);
    }
    [data-testid="stSidebar"] .stRadio label { color: #b0c4de !important; }
    [data-testid="stSidebar"] .stRadio label:hover { color: #00d4ff !important; }
    .sidebar-badge {
        display: inline-block; background: linear-gradient(135deg, #0096c7, #00d4ff);
        color: #0c1929; font-size: 0.7rem; font-weight: 700; padding: 0.2rem 0.5rem;
        border-radius: 6px; margin-top: 0.5rem;
    }
    
    /* Hero header */
    .hero {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(0, 150, 199, 0.05) 100%);
        border: 1px solid rgba(0, 212, 255, 0.25);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.08);
        animation: fadeIn 0.6s ease-out;
    }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(-8px); } to { opacity: 1; transform: translateY(0); } }
    .hero h1 { color: #00d4ff !important; font-weight: 600 !important; font-size: 1.8rem !important; margin-bottom: 0.3rem !important; }
    .hero p { color: #8ba3c7 !important; font-size: 0.95rem !important; margin: 0 !important; }
    
    /* Project title block - Home page */
    .project-title-block {
        text-align: center; padding: 2.5rem 2rem; margin-bottom: 2rem;
        background: linear-gradient(145deg, rgba(0, 212, 255, 0.06) 0%, rgba(0, 100, 140, 0.03) 100%);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 20px;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.2);
    }
    .project-title-block h1 { color: #00d4ff !important; font-size: 2rem !important; font-weight: 700 !important; margin-bottom: 0.5rem !important; }
    .project-title-block .sub { color: #6b8cae !important; font-size: 1rem !important; }
    .project-title-block .badge { display: inline-block; background: rgba(0, 212, 255, 0.2); color: #00d4ff; font-size: 0.75rem; font-weight: 600; padding: 0.35rem 0.8rem; border-radius: 20px; margin-top: 0.8rem; }
    
    /* Info cards - About */
    .info-card {
        background: linear-gradient(145deg, rgba(15, 35, 55, 0.85) 0%, rgba(10, 22, 40, 0.9) 100%);
        border: 1px solid rgba(0, 212, 255, 0.15);
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .info-card:hover { transform: translateY(-2px); box-shadow: 0 8px 28px rgba(0, 212, 255, 0.1); }
    .info-card h4 { color: #00d4ff !important; font-size: 1rem !important; margin-bottom: 0.5rem !important; }
    .info-card p { color: #b0c4de !important; font-size: 0.9rem !important; margin: 0 !important; line-height: 1.5; }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, rgba(15, 35, 55, 0.9) 0%, rgba(10, 22, 40, 0.95) 100%);
        border: 1px solid rgba(0, 212, 255, 0.12);
        border-radius: 14px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 28px rgba(0, 212, 255, 0.12); }
    .metric-card .label { font-size: 0.85rem; color: #8ba3c7; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.25rem; }
    .metric-card .value { font-size: 2rem; font-weight: 700; }
    .metric-card.good .value { color: #2ecc71; }
    .metric-card.moderate .value { color: #f1c40f; }
    .metric-card.poor .value { color: #e67e22; }
    .metric-card.unsafe .value { color: #e74c3c; }
    
    /* Big quality badge - prediction result */
    .quality-badge-large {
        display: inline-block; padding: 0.8rem 1.8rem; border-radius: 14px;
        font-size: 1.4rem; font-weight: 700; text-align: center; margin: 0.5rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    .quality-badge-large.good { background: rgba(46, 204, 113, 0.2); color: #2ecc71; border: 1px solid #2ecc71; }
    .quality-badge-large.moderate { background: rgba(241, 196, 15, 0.2); color: #f1c40f; border: 1px solid #f1c40f; }
    .quality-badge-large.poor { background: rgba(230, 126, 34, 0.2); color: #e67e22; border: 1px solid #e67e22; }
    .quality-badge-large.unsafe { background: rgba(231, 76, 60, 0.2); color: #e74c3c; border: 1px solid #e74c3c; }
    
    /* Section titles */
    h2, h3 { color: #00d4ff !important; font-weight: 600 !important; }
    p, .stMarkdown { color: #b0c4de !important; }
    
    .quality-good { color: #2ecc71 !important; font-weight: 600; font-size: 1.2rem; }
    .quality-moderate { color: #f1c40f !important; font-weight: 600; font-size: 1.2rem; }
    .quality-poor { color: #e67e22 !important; font-weight: 600; font-size: 1.2rem; }
    .quality-unsafe { color: #e74c3c !important; font-weight: 600; font-size: 1.2rem; }
    
    .result-card {
        background: linear-gradient(145deg, rgba(15, 35, 55, 0.95) 0%, rgba(10, 22, 40, 0.98) 100%);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 14px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.25);
    }
    .suggestion-box {
        padding: 1rem 1.25rem;
        border-radius: 10px;
        background: rgba(0, 212, 255, 0.06);
        border-left: 4px solid #00d4ff;
        margin: 0.6rem 0;
        color: #b0c4de;
        font-size: 0.95rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #0096c7 0%, #00d4ff 100%) !important;
        color: #0c1929 !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.5rem !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.4) !important;
    }
    
    .stDataFrame { border-radius: 12px; overflow: hidden; }
    div[data-testid="stMetricValue"] { font-size: 1.9rem; font-weight: 700; }
    
    .footer-note { color: #5a7a9e !important; font-size: 0.75rem !important; margin-top: 1.5rem !important; text-align: center; }
    
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

def load_all_data():
    """Load and merge train + test for dashboard."""
    dfs = []
    for folder, name in [(TRAIN_DIR, "train.csv"), (TEST_DIR, "test.csv")]:
        path = os.path.join(folder, name)
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)

def render_sidebar():
    with st.sidebar:
        st.markdown("### 💧 Water Quality Monitor")
        
        st.markdown("Data-Driven Modeling & Visualization of Water Quality parameters")
        st.markdown("---")
        page = st.radio(
            "*Navigate*",
            ["About Project", "Dashboard", "Input & Predict", "City-wise Quality", "Upload Data"],
            index=0,
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.caption("Python • Streamlit • Scikit-learn")
        
    return page

def page_home():
    """About Project - Final year project overview."""
    st.markdown("""
    <div class="project-title-block">
        <h3>💧 Data-Driven Modeling & Visualization of Water Quality parameters</h3>
        <p class="sub">Intelligent prediction and city-wise monitoring using Machine Learning</p>
        <span class="badge">Computer Science / IT</span>
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>📋 Abstract</h4>
            <p>This project presents an NLP-based Water Quality Monitoring System that classifies water quality from multiple physicochemical parameters (pH, turbidity, dissolved oxygen, TDS, nitrate, etc.) and provides actionable suggestions. The system supports city-wise analytics, manual input prediction, and CSV upload for bulk analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
            <h4>🎯 Objectives</h4>
            <p>• Classify water quality into Good, Moderate, Poor, and Unsafe.<br>
            • Provide real-time prediction and improvement suggestions from user input.<br>
            • Visualize city-wise quality distribution and trends.<br>
            • Allow data upload and integration for continuous learning.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>⚙️ Methodology</h4>
            <p>Data is generated/uploaded with 9 key parameters. A Random Forest classifier is trained on labeled data (80% train, 20% test). The Streamlit dashboard offers: Dashboard (metrics & charts), Input & Predict (single-sample prediction + suggestions), City-wise Quality (geographic breakdown), and Upload Data (CSV import).</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
            <h4>🛠️ Tech Stack</h4>
            <p>Python, Streamlit (UI), Pandas (data), Scikit-learn (Random Forest), Plotly (visualization), Joblib (model persistence). Quality thresholds aligned with WHO/EPA guidelines.</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Quick stats")
    df = load_all_data()
    if df is not None and not df.empty:
        total = len(df)
        cities = df["city"].nunique()
        good_pct = round(100 * (df["quality"] == "Good").sum() / total, 1)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total samples", f"{total:,}")
        c2.metric("Cities monitored", cities)
        c3.metric("Good quality %", f"{good_pct}%")
    else:
        st.info("Run the app with data (train/test CSV) to see quick stats here.")
    with st.expander("📌 Future scope & references"):
        st.markdown("""
        *Future scope:* Integration with IoT sensors for real-time data, mobile app, API for third-party systems, multi-language support, and historical trend analysis.
        *References:* WHO Guidelines for Drinking-water Quality; EPA Water Quality Standards; Scikit-learn Documentation; Streamlit Documentation.
        """)

def page_dashboard(df):
    st.markdown('<div class="hero"><h1>📊 Overview Dashboard</h1><p>Real-time water quality metrics across all monitored locations</p></div>', unsafe_allow_html=True)
    if df is None or df.empty:
        st.warning("No data found. Run data_generator.py and add data to train/test folders.")
        return
    total = len(df)
    cities = df["city"].nunique()
    st.caption(f"Total samples: *{total:,}*  •  Cities: *{cities}*")
    c1, c2, c3, c4 = st.columns(4)
    for col, (label, color) in zip([c1, c2, c3, c4], QUALITY_COLORS.items()):
        with col:
            count = (df["quality"] == label).sum()
            cls = label.lower()
            st.markdown(f'<div class="metric-card {cls}"><div class="label">{label}</div><div class="value">{count}</div></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Quality distribution")
    fig = px.pie(df, names="quality", color="quality", color_discrete_map=QUALITY_COLORS, title="")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#b0c4de", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Quality by parameter (averages)")
    num_cols = ["ph", "turbidity", "dissolved_oxygen", "total_dissolved_solids", "nitrate"]
    avg_by_quality = df.groupby("quality")[num_cols].mean().round(2)
    st.dataframe(avg_by_quality, use_container_width=True, hide_index=True)
    st.subheader("Recent samples (last 50)")
    st.dataframe(df.tail(50), use_container_width=True, hide_index=True)

def page_input_predict():
    st.markdown('<div class="hero"><h1>🔬 Input & Predict</h1><p>Enter water parameters to get quality prediction and improvement suggestions</p></div>', unsafe_allow_html=True)
    with st.form("input_form"):
        st.markdown("*Water quality parameters*")
        c1, c2 = st.columns(2)
        with c1:
            ph = st.slider("pH", 4.0, 10.0, 7.0, 0.1, help="Ideal: 6.5–8.5")
            turbidity = st.number_input("Turbidity (NTU)", 0.0, 100.0, 5.0, 0.1, help="Lower is better")
            dissolved_oxygen = st.number_input("Dissolved Oxygen (mg/L)", 0.0, 20.0, 8.0, 0.1, help="Higher is better")
            temperature = st.number_input("Temperature (°C)", 0.0, 40.0, 25.0, 0.1)
        with c2:
            conductivity = st.number_input("Conductivity (µS/cm)", 0.0, 3000.0, 500.0, 10.0)
            tds = st.number_input("Total Dissolved Solids (mg/L)", 0.0, 2500.0, 300.0, 10.0)
            nitrate = st.number_input("Nitrate (mg/L)", 0.0, 50.0, 5.0, 0.5, help="Safe: &lt; 10 mg/L")
            fecal_coliform = st.number_input("Fecal Coliform (count/100mL)", 0, 500, 0, 1)
            bod = st.number_input("BOD (mg/L)", 0.0, 30.0, 2.0, 0.1)
        submitted = st.form_submit_button("🔍 Predict Quality")
    if submitted:
        params = {
            "ph": ph, "turbidity": turbidity, "dissolved_oxygen": dissolved_oxygen,
            "temperature": temperature, "conductivity": conductivity,
            "total_dissolved_solids": tds, "nitrate": nitrate,
            "fecal_coliform": fecal_coliform, "bod": bod,
        }
        label, suggestions = predict_quality(params)
        if label is None:
            st.error("Model not loaded. Run train_model.py first.")
        else:
            css_class = f"quality-{label.lower()}"
            icon = {"Good": "✅", "Moderate": "⚠️", "Poor": "🔶", "Unsafe": "🚫"}.get(label, "")
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(f"*Predicted quality*")
            st.markdown(f'<div class="quality-badge-large {css_class}">{icon} {label}</div>', unsafe_allow_html=True)
            st.markdown("*Recommendations*")
            for s in suggestions:
                st.markdown(f'<div class="suggestion-box">💡 {s}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

def page_city_wise(df):
    st.markdown('<div class="hero"><h1>🏙️ City-wise Quality</h1><p>Quality breakdown and dominant status per city</p></div>', unsafe_allow_html=True)
    if df is None or df.empty:
        st.warning("No data. Add data to train/test folders and reload.")
        return
    city_quality = df.groupby("city")["quality"].value_counts().unstack(fill_value=0)
    city_quality = city_quality.reindex(columns=QUALITY_LABELS, fill_value=0)
    fig = px.bar(
        city_quality.reset_index().melt(id_vars="city", var_name="Quality", value_name="Count"),
        x="city", y="Count", color="Quality", barmode="stack",
        color_discrete_map=QUALITY_COLORS, title="",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 35, 55, 0.4)",
        font=dict(color="#b0c4de", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        xaxis=dict(tickangle=-45, gridcolor="rgba(0,212,255,0.1)"),
        yaxis=dict(gridcolor="rgba(0,212,255,0.1)"),
        margin=dict(t=60, b=80),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Dominant quality by city")
    dominant = df.groupby("city")["quality"].agg(lambda x: x.mode().iloc[0] if len(x) else "N/A").reset_index()
    dominant.columns = ["City", "Dominant Quality"]
    st.dataframe(dominant, use_container_width=True, hide_index=True)

def page_upload():
    st.markdown('<div class="hero"><h1>📁 Upload Data</h1><p>Import CSV to view or add to the training set</p></div>', unsafe_allow_html=True)
    st.info("Upload a CSV with columns: *city, **ph, **turbidity, **dissolved_oxygen, **temperature, **conductivity, **total_dissolved_solids, **nitrate, **fecal_coliform, **bod* (optional: *quality*).")
    file = st.file_uploader("Choose CSV", type=["csv"])
    if file:
        try:
            up = pd.read_csv(file)
            required = ["city"] + FEATURES
            missing = [c for c in required if c not in up.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                st.success(f"Loaded {len(up)} rows.")
                st.dataframe(up.head(20), use_container_width=True, hide_index=True)
                if "quality" not in up.columns and st.button("Add to training set (append to train.csv)"):
                    model, le = load_model()
                    if model is not None and le is not None:
                        preds = model.predict(up[FEATURES])
                        up["quality"] = le.inverse_transform(preds)
                    else:
                        from data_generator import compute_quality_label
                        up["quality"] = up.apply(compute_quality_label, axis=1)
                    train_path = os.path.join(TRAIN_DIR, "train.csv")
                    os.makedirs(TRAIN_DIR, exist_ok=True)
                    if os.path.exists(train_path):
                        existing = pd.read_csv(train_path)
                        combined = pd.concat([existing, up], ignore_index=True)
                        combined.to_csv(train_path, index=False)
                        st.success("Appended to train.csv. Re-run training to update model.")
                    else:
                        up.to_csv(train_path, index=False)
                        st.success("Saved as train.csv.")
        except Exception as e:
            st.exception(e)

def main():
    page = render_sidebar()
    df = load_all_data()
    if page == "About Project":
        page_home()
    elif page == "Dashboard":
        page_dashboard(df)
    elif page == "Input & Predict":
        page_input_predict()
    elif page == "City-wise Quality":
        page_city_wise(df)
    else:
        page_upload()

if _name_ == "_main_":
    main()