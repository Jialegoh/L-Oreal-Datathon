import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import os

# Page configuration
st.set_page_config(page_title="AI Glow-rithms", layout="wide")

st.title("Dashboard")
st.markdown("Analyse the quality and relevance of comments through Share of Engagement (SoE)")

# Load data from fixed CSV path (no file-type branching)
base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, "comments_with_sentiment.csv")
try:
    df = pd.read_csv(data_path)
except Exception as e:
    st.error(f"Failed to read {data_path}: {e}")
    st.stop()

# Optional: compute total_engagement if columns exist
if set(["likes", "shares", "saves", "comments"]).issubset(df.columns):
    df["total_engagement"] = df[["likes","shares","saves","comments"]].sum(axis=1)

# Sidebar 
st.sidebar.header("Filters")
if "post_id" in df.columns:
    post_filter = st.sidebar.multiselect("Select Post ID(s):", options=df["post_id"].unique(), default=df["post_id"].unique())
    df = df[df["post_id"].isin(post_filter)]
if "sentiment" in df.columns:
    sentiment_filter = st.sidebar.multiselect("Select Sentiment(s):", options=df["sentiment"].unique(), default=df["sentiment"].unique())
    df = df[df["sentiment"].isin(sentiment_filter)]
if "quality_score" in df.columns:
    quality_filter = st.sidebar.slider("Quality Score Range", 0.0, 1.0, (0.0, 1.0))
    df = df[(df["quality_score"] >= quality_filter[0]) & (df["quality_score"] <= quality_filter[1])]

# Adding tab for better orgnasation
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Sentiment", "Trends", "WordCloud", "AI Model Predictions"])

# tab 1: Overview
with tab1:
    st.subheader("Key Metrics")

    if set(["likes", "shares", "saves", "comments"]).issubset(df.columns):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("👍 Total Likes", int(df["likes"].sum()))
        col2.metric("🔁 Total Shares", int(df["shares"].sum()))
        col3.metric("💾 Total Saves", int(df["saves"].sum()))
        col4.metric("💬 Total Comments", int(df["comments"].sum()))

        # Engagement Breakdown
        engagement_cols = ["likes", "shares", "saves", "comments"]
        engagement_sum = df[engagement_cols].sum().reset_index()
        engagement_sum.columns = ["metric", "count"]
        fig1 = px.bar(engagement_sum, x="metric", y="count", title="Engagement Breakdown", text="count")
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("Engagement metrics require likes/shares/saves/comments columns.")

# tab 2 : Sentiment Analysis
with tab2:
    st.subheader("Sentiment Analysis")
    if "sentiment" in df.columns:
        vc = df["sentiment"].astype(str).str.lower().value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("Positive", int(vc.get("positive", 0)))
        c2.metric("Neutral", int(vc.get("neutral", 0)))
        c3.metric("Negative", int(vc.get("negative", 0)))

        fig2 = px.histogram(df, x="sentiment", color="sentiment",
                            title="Sentiment Distribution", text_auto=True)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No 'sentiment' column found to summarize and plot.")

    if set(["quality_score", "total_engagement", "sentiment"]).issubset(df.columns):
        st.subheader("Comment Quality vs Engagement")
        fig3 = px.scatter(df, x="quality_score", y="total_engagement",
                          color="sentiment", hover_data=["post_id","comment_id"],
                          size="likes", size_max=15, title="Quality vs Engagement")
        st.plotly_chart(fig3, use_container_width=True)

# tab 3: Trends
with tab3:
    if set(["timestamp", "total_engagement"]).issubset(df.columns):
        st.subheader("Engagement Over Time")
        fig4 = px.line(df.groupby("timestamp")[["total_engagement"]].sum().reset_index(),
                       x="timestamp", y="total_engagement", markers=True,
                       title="Total Engagement Trend")
        st.plotly_chart(fig4, use_container_width=True)

    if set(["timestamp", "quality_score"]).issubset(df.columns):
        st.subheader("Average Quality Over Time")
        fig5 = px.line(df.groupby("timestamp")[["quality_score"]].mean().reset_index(),
                       x="timestamp", y="quality_score", markers=True,
                       title="Quality Score Trend")
        st.plotly_chart(fig5, use_container_width=True)

# tab 4: WordCloud
with tab4:
    st.subheader("Word Cloud of Comments")
    text = "positive helpful great amazing love bad awful toxic fun engaging high-quality insightful"
    wordcloud = WordCloud(width=400, height=100, background_color="white").generate(text)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)
