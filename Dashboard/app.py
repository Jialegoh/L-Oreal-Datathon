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


st.subheader("📌 Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Likes", int(df["likes"].sum()))
col2.metric("Total Shares", int(df["shares"].sum()))
col3.metric("Total Saves", int(df["saves"].sum()))
col4.metric("Total Comments", int(df["comments"].sum()))

# Engagement Breakdown
st.subheader("Engagement Breakdown")
engagement_cols = ["likes", "shares", "saves", "comments"]
engagement_sum = df[engagement_cols].sum().reset_index()
engagement_sum.columns = ["metric", "count"]
fig1 = px.bar(engagement_sum, x="metric", y="count", title="Total Engagement Metrics")
st.plotly_chart(fig1, use_container_width=True)

# Quality vs Engagement
st.subheader("Comment Quality vs Engagement")
fig2 = px.scatter(df, x="quality_score", y="total_engagement",
                  color="sentiment", hover_data=["post_id","comment_id"])
st.plotly_chart(fig2, use_container_width=True)

# Sentiment Distribution
st.subheader("Sentiment Distribution")
fig3 = px.histogram(df, x="sentiment", color="sentiment", title="Distribution of Comment Sentiments")
st.plotly_chart(fig3, use_container_width=True)

# Engagement Over Time
st.subheader("Engagement Over Time")
fig4 = px.line(df.groupby("timestamp")[["total_engagement"]].sum().reset_index(),
               x="timestamp", y="total_engagement", markers=True)
st.plotly_chart(fig4, use_container_width=True)

# Word Cloud (Mock Example)
st.subheader("Word Cloud of Comments (Mock)")
text = "positive helpful great amazing love bad awful toxic fun engaging high-quality"
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
st.pyplot(plt)
