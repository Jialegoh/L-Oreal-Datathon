import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="AI Glow-rithms", layout="wide")

st.title("Dashboard")
st.markdown("Analyse the quality and relevance of comments through Share of Engagement (SoE)")

# Load data
df = pd.DataFrame({
    "comment_id": range(1, 11),
    "post_id": [101,102,101,103,104,102,105,101,106,107],
    "likes": [20, 50, 10, 5, 40, 12, 70, 22, 8, 15],
    "shares": [5, 15, 2, 1, 6, 2, 20, 3, 0, 5],
    "saves": [2, 10, 1, 0, 4, 1, 15, 2, 1, 3],
    "comments": [10, 25, 5, 3, 12, 4, 30, 6, 2, 7],
    "sentiment": ["positive","negative","neutral","positive","positive",
                  "negative","positive","neutral","negative","positive"],
    "quality_score": [0.8,0.3,0.6,0.9,0.7,0.4,0.95,0.5,0.2,0.85],
    "timestamp": pd.date_range("2024-01-01", periods=10, freq="D")
})
df["total_engagement"] = df[["likes","shares","saves","comments"]].sum(axis=1)


# Sidebar 
st.sidebar.header("Filters")

# Filter by Post
post_filter = st.sidebar.multiselect("Select Post ID(s):", options=df["post_id"].unique(), default=df["post_id"].unique())
df = df[df["post_id"].isin(post_filter)]

# Filter by Sentiment
sentiment_filter = st.sidebar.multiselect("Select Sentiment(s):", options=df["sentiment"].unique(), default=df["sentiment"].unique())
df = df[df["sentiment"].isin(sentiment_filter)]

# Filter by Date Range
date_min, date_max = df["timestamp"].min(), df["timestamp"].max()
date_range = st.sidebar.date_input("Select Date Range:", [date_min, date_max])
if len(date_range) == 2:
    start_date, end_date = date_range
    df = df[(df["timestamp"] >= pd.to_datetime(start_date)) & (df["timestamp"] <= pd.to_datetime(end_date))]

# Adding tab for better orgnasation
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Sentiment", "Trends", "WordCloud"])

# tab 1: Overview
with tab1:
    st.subheader("📌 Key Metrics")

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

# tab 2 : Sentiment Analysis
with tab2:
    st.subheader("😊 Sentiment Analysis")
    fig2 = px.histogram(df, x="sentiment", color="sentiment",
                        title="Sentiment Distribution", text_auto=True)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Comment Quality vs Engagement")
    fig3 = px.scatter(df, x="quality_score", y="total_engagement",
                      color="sentiment", hover_data=["post_id","comment_id"],
                      size="likes", size_max=15, title="Quality vs Engagement")
    st.plotly_chart(fig3, use_container_width=True)

# tab 3: Trends
with tab3:
    st.subheader("📈 Engagement Over Time")
    fig4 = px.line(df.groupby("timestamp")[["total_engagement"]].sum().reset_index(),
                   x="timestamp", y="total_engagement", markers=True,
                   title="Total Engagement Trend")
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Average Quality Over Time")
    fig5 = px.line(df.groupby("timestamp")[["quality_score"]].mean().reset_index(),
                   x="timestamp", y="quality_score", markers=True,
                   title="Quality Score Trend")
    st.plotly_chart(fig5, use_container_width=True)

# tab 4: WordCloud
with tab4:
    st.subheader("☁ Word Cloud of Comments (Mock)")
    text = "positive helpful great amazing love bad awful toxic fun engaging high-quality insightful"
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)