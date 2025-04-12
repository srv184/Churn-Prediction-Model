import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from fpdf import FPDF
import base64

# Set page configuration at the very top
st.set_page_config(page_title="Advanced Churn Analysis App", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #1e1e2f;
            color: #fff;
            font-family: 'Roboto', sans-serif;
        }

        h1 {
            font-size: 2.5em;
            color: #8e44ad;
            text-align: center;
            margin-top: 30px;
            text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
        }

        .sidebar .sidebar-content {
            background-color: #2c3e50;
        }

        .stMarkdown {
            font-size: 1.2em;
            font-weight: 500;
            color: #ecf0f1;
            margin-top: -15px;
        }

        .stMetric {
            font-size: 1.25em;
            background-color: #34495e;
            padding: 10px;
            border-radius: 12px;
            box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.3);
        }

        .stMetric .stMetricLabel {
            font-weight: bold;
            color: #d35400;
        }

        .stMetric .stMetricValue {
            color: #ecf0f1;
        }

        .stButton>button {
            background-color: #8e44ad;
            color: white;
            border-radius: 50px;
            padding: 12px 24px;
            border: none;
            font-size: 1.1em;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }

        .stButton>button:hover {
            background-color: #9b59b6;
        }

        .stPlotlyChart {
            border-radius: 12px;
            box-shadow: 0px 0px 20px rgba(0, 0, 0, 0.3);
        }

        .stHeader {
            background-color: #2c3e50;
            color: #fff;
            padding: 30px 0;
        }

        .stSidebar .sidebar-content {
            background-color: #34495e;
        }
        
        footer {
            text-align: center;
            color: #95a5a6;
            font-size: 0.9em;
            margin-top: 50px;
        }
    </style>
""", unsafe_allow_html=True)

# Page Configuration
st.title("📊 Advanced Churn Analysis Dashboard")
st.markdown("Upload your customer data for churn prediction, visualization, and actionable insights.")

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("📁 Upload a CSV file", type=["csv"])

# --- DATA ENRICHMENT ---
def enrich_data(df):
    df = df.copy()
    df['tenure'] = pd.to_numeric(df.get('tenure', pd.Series([0]*len(df))), errors='coerce')
    df['monthly_charges'] = pd.to_numeric(df.get('monthly_charges', pd.Series([0]*len(df))), errors='coerce')

    if 'total_charges' not in df.columns:
        df['total_charges'] = df['monthly_charges'] * df['tenure']

    df['long_term_customer'] = df['tenure'] > 12
    df['high_value_customer'] = df['total_charges'] > 2000
    df['recent_low_value_customer'] = (df['tenure'] < 12) & (df['total_charges'] < 1000)
    df['CLV'] = df['monthly_charges'] * df['tenure']

    return df

# --- PDF EXPORT ---
def generate_pdf_report(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Churn Analysis Report", ln=True, align="C")

    pdf.cell(200, 10, txt=f"Total Customers: {len(df)}", ln=True)
    pdf.cell(200, 10, txt=f"Churned: {(df['churn_prediction']=='Yes').sum()}", ln=True)
    pdf.cell(200, 10, txt=f"Retention: {(df['churn_prediction']=='No').sum()}", ln=True)

    file_path = "churn_report.pdf"
    pdf.output(file_path)
    return file_path

# --- MAIN LOGIC ---
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df.drop_duplicates(inplace=True)
    df.dropna(how='all', inplace=True)
    df.dropna(thresh=len(df.columns) // 2, inplace=True)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    st.success("✅ Data Uploaded and Cleaned")
    enriched_df = enrich_data(df)

    # --- Churn Prediction ---
    enriched_df['churn_prediction'] = np.where(
        (enriched_df['tenure'] < 12) & (enriched_df['total_charges'] < 1000), 'Yes', 'No'
    )

    # --- Key Performance Indicators ---
    st.header("📌 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    churn_rate = (enriched_df['churn_prediction'] == 'Yes').mean() * 100
    retention_rate = 100 - churn_rate
    avg_ltv = enriched_df['CLV'].mean()
    arpu = enriched_df['monthly_charges'].mean()

    col1.metric("Churn Rate", f"{churn_rate:.2f}%")
    col2.metric("Retention Rate", f"{retention_rate:.2f}%")
    col3.metric("Avg. Customer LTV", f"${avg_ltv:,.2f}")
    col4.metric("ARPU", f"${arpu:,.2f}")

    # --- Visualizations ---
    st.header("📊 Visualizations")
    st.subheader("Churn Distribution")
    fig1 = px.histogram(enriched_df, x="churn_prediction", color="churn_prediction", barmode='group')
    st.plotly_chart(fig1, use_container_width=True)

    if 'tenure' in enriched_df.columns:
        st.subheader("Churn Over Tenure")
        fig2 = px.line(enriched_df, x='tenure', y='CLV', color='churn_prediction', markers=True)
        st.plotly_chart(fig2, use_container_width=True)

    if 'monthly_charges' in enriched_df.columns:
        st.subheader("Monthly Charges Boxplot by Churn")
        fig3 = px.box(enriched_df, x="churn_prediction", y="monthly_charges", color="churn_prediction")
        st.plotly_chart(fig3, use_container_width=True)

    if 'total_charges' in enriched_df.columns:
        st.subheader("Customer Value Distribution")
        scatter_df = enriched_df.dropna(subset=['monthly_charges'])  # Fix: remove NaNs
        fig4 = px.scatter(
            scatter_df,
            x="tenure",
            y="total_charges",
            color="churn_prediction",
            size="monthly_charges"
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("📌 Feature Correlation Heatmap")
    corr_df = enriched_df.select_dtypes(include=np.number).corr()
    fig5, ax5 = plt.subplots()
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", ax=ax5)
    st.pyplot(fig5)

    st.subheader("📈 Retention vs Churn Pie")
    churn_counts = enriched_df['churn_prediction'].value_counts()
    fig6 = px.pie(names=churn_counts.index, values=churn_counts.values, title="Churn vs Retention")
    st.plotly_chart(fig6)

    # --- PDF REPORT ---
    st.subheader("📤 Export Report")
    if st.button("Download PDF Report"):
        file_path = generate_pdf_report(enriched_df)
        with open(file_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="churn_report.pdf">📥 Download Report</a>'
            st.markdown(href, unsafe_allow_html=True)

    # --- CSV DOWNLOAD ---
    st.download_button("📥 Download Enriched Data", data=enriched_df.to_csv(index=False), file_name="churn_enriched.csv", mime="text/csv")

else:
    st.info("📂 Please upload a CSV file to begin.")
