import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="RFM Analysis Dashboard", layout="wide")

st.title("📊 RFM Customer Segmentation Dashboard")

# 1. File Upload
uploaded_file = st.file_uploader("Upload a CSV file with transaction data", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Raw Data Preview")
    st.dataframe(df.head())

    # 2. Preprocessing
    st.subheader("⚙️ Data Cleaning & Preparation")

    # Ensure required columns exist
    required_cols = {"InvoiceNo", "CustomerID", "InvoiceDate", "Quantity", "UnitPrice"}
    if not required_cols.issubset(df.columns):
        st.error(f"Missing required columns: {required_cols - set(df.columns)}")
    else:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
        df.dropna(subset=["CustomerID"], inplace=True)
        df["CustomerID"] = df["CustomerID"].astype(int)

        # 3. RFM Calculation
        st.subheader("📊 RFM Calculation")
        ref_date = df["InvoiceDate"].max() + timedelta(days=1)

        rfm = df.groupby("CustomerID").agg({
            "InvoiceDate": lambda x: (ref_date - x.max()).days,
            "InvoiceNo": "nunique",
            "TotalPrice": "sum"
        }).rename(columns={
            "InvoiceDate": "Recency",
            "InvoiceNo": "Frequency",
            "TotalPrice": "Monetary"
        }).reset_index()

        # 4. RFM Scoring
        rfm["R_Score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1])
        rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
        rfm["M_Score"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5])
        rfm["RFM_Segment"] = rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)
        rfm["RFM_Score"] = rfm[["R_Score", "F_Score", "M_Score"]].astype(int).sum(axis=1)

        # Segmenting customers based on score
        def segment(score):
            if score >= 13:
                return "Loyal"
            elif score >= 9:
                return "Potential"
            elif score >= 6:
                return "Needs Attention"
            else:
                return "At Risk"

        rfm["Segment"] = rfm["RFM_Score"].apply(segment)

        st.dataframe(rfm.head())

        # 5. Visualizations
        st.subheader("📈 Visualizations")

        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.histogram(rfm, x="Segment", color="Segment", title="Customer Segments Count")
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.scatter(rfm, x="Recency", y="Monetary", size="Frequency", color="Segment",
                              title="Recency vs Monetary (bubble size = Frequency)", hover_data=["CustomerID"])
            st.plotly_chart(fig2, use_container_width=True)

        # 6. Download Segment Data
        st.subheader("📥 Download RFM Data")
        csv = rfm.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv, file_name="rfm_analysis.csv", mime="text/csv")
