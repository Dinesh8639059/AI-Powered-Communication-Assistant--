# src/analytics.py
import pandas as pd
import streamlit as st

def get_stats(df: pd.DataFrame) -> dict:
    """Compute basic and extended analytics"""
    now = pd.Timestamp.now()
    last_24h = df[df.get("Sent Date", pd.NaT) > (now - pd.Timedelta(days=1))]
    
    stats = {
        "Total Emails": len(df),
        "Last 24h": len(last_24h),
        "Urgent": int((df["Priority"] == "Urgent").sum()),
        "High Priority": int((df["Priority"] == "High").sum()),
        "Normal": int((df["Priority"] == "Normal").sum()),
        "Positive": int((df["Sentiment"] == "Positive").sum()),
        "Negative": int((df["Sentiment"] == "Negative").sum()),
        "Neutral": int((df["Sentiment"] == "Neutral").sum()),
    }
    return stats

def show_charts(df: pd.DataFrame):
    st.write("### Sentiment Distribution")
    st.bar_chart(df["Sentiment"].value_counts())

    st.write("### Priority Distribution")
    st.bar_chart(df["Priority"].value_counts())

    st.write("### Requirement Categories")
    if "Requirement" in df.columns:
        st.bar_chart(df["Requirement"].value_counts())

    st.write("### Top 5 Senders")
    if "From" in df.columns:
        top_senders = df["From"].value_counts().head(5)
        st.bar_chart(top_senders)

    st.write("### Emails Over Last 7 Days")
    if "Sent Date" in df.columns:
        df_dates = df.copy()
        df_dates["Sent Date"] = pd.to_datetime(df_dates["Sent Date"], errors="coerce")
        last_7 = df_dates[df_dates["Sent Date"] >= (pd.Timestamp.now() - pd.Timedelta(days=7))]
        if not last_7.empty:
            count_by_day = last_7.groupby(last_7["Sent Date"].dt.date).size()
            st.line_chart(count_by_day)
