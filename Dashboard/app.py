import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import os
import json
import numpy as np

# Page configuration
st.set_page_config(page_title="AI Glow-rithms", layout="wide")

st.title("Dashboard")
st.markdown("Analyse the quality and relevance of comments through Share of Engagement (SoE)")

# Load data from fixed CSV path
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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Overview", "Sentiment", "Trends", "WordCloud", "Predictions", "Classification Model"])

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

# tab 6: Classification Model (BERT multi-label)
with tab6:
    st.subheader("Classification Model")
    st.markdown("This module classifies user comments into multiple topic categories using a fine-tuned BERT model.")

    # Paths
    dashboard_dir = os.path.dirname(__file__)
    model_dir = os.path.normpath(os.path.join(dashboard_dir, "..", "AI_Model", "CommentCategory", "saved_model"))

    # Cache heavy objects
    @st.cache_resource(show_spinner=False)
    def load_model_bundle(saved_path: str):
        try:
            import torch  # local import to avoid loading if unused
            from transformers import BertTokenizer, BertForSequenceClassification
            tokenizer = BertTokenizer.from_pretrained(saved_path)
            model = BertForSequenceClassification.from_pretrained(saved_path)
            model.eval()
            return model, tokenizer
        except Exception as e:
            st.warning(f"Failed to load model from {saved_path}: {e}")
            return None, None

    @st.cache_data(show_spinner=False)
    def load_classes(saved_path: str):
        """Load label list from mlb_classes.npy, or fallback to config.json id2label."""
        classes_path = os.path.join(saved_path, "mlb_classes.npy")
        if os.path.exists(classes_path):
            try:
                classes = np.load(classes_path, allow_pickle=True)
                return classes.tolist()
            except Exception as e:
                st.warning(f"Failed to load classes from numpy file: {e}")
        # Fallback to config.json
        cfg_path = os.path.join(saved_path, "config.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                id2label = cfg.get("id2label") or {}
                # Ensure order by id index
                if isinstance(id2label, dict):
                    return [id2label.get(str(i), str(i)) for i in range(len(id2label))]
            except Exception as e:
                st.warning(f"Failed to parse labels from config.json: {e}")
        return []

    @st.cache_data(show_spinner=False)
    def load_optional_json(path: str):
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    classes = load_classes(model_dir)
    model, tokenizer = load_model_bundle(model_dir)

    # Metrics snapshot: prefer overall_classification_metrics.csv, else metrics.json, else classification_report.csv, else trainer_state.json
    metrics = {}
    overall_path = os.path.join(model_dir, "overall_classification_metrics.csv")
    if os.path.exists(overall_path):
        try:
            overall_df = pd.read_csv(overall_path)
            # Normalize names
            def get_row(label):
                row = overall_df[overall_df["average_type"].str.lower()==label].iloc[0]
                return row
            micro = get_row("micro avg")
            macro = get_row("macro avg")
            weighted = get_row("weighted avg")
            metrics = {
                "micro_f1": float(micro["f1_score"]),
                "macro_f1": float(macro["f1_score"]),
                "weighted_f1": float(weighted["f1_score"]),
                "precision": float(micro["precision"]),
                "recall": float(micro["recall"]),
            }
        except Exception:
            metrics = {}
    if not metrics:
        metrics = load_optional_json(os.path.join(model_dir, "metrics.json")) or {}
    # If we have per-label CSV, compute fallback macro/weighted metrics from it
    clsrep_path = os.path.join(model_dir, "classification_report.csv")
    per_label_from_clsrep = None
    if os.path.exists(clsrep_path):
        try:
            per_label_from_clsrep = pd.read_csv(clsrep_path)
            if {"precision","recall","f1-score"}.issubset(per_label_from_clsrep.columns):
                metrics.setdefault("macro_f1", float(per_label_from_clsrep["f1-score"].mean()))
                if "support" in per_label_from_clsrep.columns:
                    w = per_label_from_clsrep["support"].astype(float)
                    w = w / (w.sum() if w.sum() else 1.0)
                    metrics.setdefault("weighted_f1", float((per_label_from_clsrep["f1-score"] * w).sum()))
        except Exception:
            per_label_from_clsrep = None
    if not metrics:
        trainer_state = load_optional_json(os.path.join(model_dir, "trainer_state.json"))
        if trainer_state:
            # best_metric often maps to eval_micro/f1 when metric_for_best_model is set accordingly
            if isinstance(trainer_state, dict):
                metrics["micro_f1"] = trainer_state.get("best_metric")
    # Graceful defaults if still missing
    metrics.setdefault("micro_f1", 0.000)
    metrics.setdefault("macro_f1", 0.000)
    metrics.setdefault("precision", 0.000)
    metrics.setdefault("recall", 0.000)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Micro-F1", f"{metrics.get('micro_f1', 0):.3f}")
    col2.metric("Macro-F1", f"{metrics.get('macro_f1', 0):.3f}")
    col3.metric("Weighted-F1", f"{metrics.get('weighted_f1', 0):.3f}")
    col4.metric("Precision", f"{metrics.get('precision', 0):.3f}")
    col5.metric("Recall", f"{metrics.get('recall', 0):.3f}")

    st.divider()

    # Dataset Overview
    st.markdown("**Dataset Overview**")
    st.markdown("The commentVideoMerged.csv file was created by merging the comment and video files on video_id and channel_id.")
    st.divider()
    


    # Per-label performance (Top 10 by frequency)
    st.markdown("**Per-Category Performance (Top 10 by frequency)**")
    per_label_f1_path = os.path.join(model_dir, "per_label_f1.csv")
    if per_label_from_clsrep is not None:
        df_show = per_label_from_clsrep.copy()
        if "support" in df_show.columns:
            df_show = df_show.sort_values("support", ascending=False).head(10)
        st.dataframe(df_show, use_container_width=True, height=320)
        try:
            fig_f1 = px.bar(df_show, x="label", y="f1-score", title="Top-10 Categories by Support — F1 Scores")
            st.plotly_chart(fig_f1, use_container_width=True)
        except Exception:
            pass
    elif os.path.exists(per_label_f1_path):
        per_label_df = pd.read_csv(per_label_f1_path)
        st.dataframe(per_label_df, use_container_width=True, height=320)
        try:
            fig_f1 = px.bar(per_label_df, x="label", y="f1", title="Per-Label F1 Scores")
            st.plotly_chart(fig_f1, use_container_width=True)
        except Exception:
            pass
    elif classes:
        st.info("No saved per-label metrics found. Provide classification_report.csv or per_label_f1.csv to visualize actual scores.")
        placeholder_df = pd.DataFrame({"label": classes, "f1": [None] * len(classes)})
        st.dataframe(placeholder_df, use_container_width=True, height=320)

    st.divider()

    # Threshold tuning results: compare default vs optimal-per-class
    st.markdown("**Threshold Tuning Results (0.5 vs optimal per class)**")
    tuned_path = os.path.join(model_dir, "classification_report_with_thresholds.csv")
    if os.path.exists(tuned_path) and os.path.exists(clsrep_path):
        try:
            df_default = pd.read_csv(clsrep_path)[["label","f1-score"]].rename(columns={"f1-score":"f1_default"})
            df_tuned = pd.read_csv(tuned_path)[["label","f1-score","optimal_threshold"]].rename(columns={"f1-score":"f1_tuned"})
            comp = df_default.merge(df_tuned, on="label", how="inner")
            comp["delta"] = comp["f1_tuned"] - comp["f1_default"]
            comp_sorted = comp.sort_values("delta", ascending=False)
            st.dataframe(comp_sorted, use_container_width=True, height=340)
            # Chart: show improvement deltas (top 10)
            fig_delta = px.bar(comp_sorted.head(10), x="label", y="delta", title="Top-10 Improvements after Threshold Tuning")
            st.plotly_chart(fig_delta, use_container_width=True)
        except Exception as e:
            st.info(f"Could not compute tuning comparison: {e}")
    else:
        st.info("Provide classification_report.csv and classification_report_with_thresholds.csv to show tuning improvements.")

    # Build thresholds map for interactive predictions
    thresholds = {}
    opt_thr_path = os.path.join(model_dir, "optimal_thresholds.csv")
    if os.path.exists(opt_thr_path):
        try:
            thr_df = pd.read_csv(opt_thr_path)
            if {"label","optimal_threshold"}.issubset(thr_df.columns):
                thresholds = dict(zip(thr_df["label"], thr_df["optimal_threshold"]))
        except Exception:
            thresholds = {}
    if not thresholds and classes:
        thresholds = {label: 0.50 for label in classes}

    # Interactive Demo
    st.markdown("**Interactive Demo**")
    user_text = st.text_area("Enter a comment to classify:", height=120, placeholder="Type or paste a comment...")

    def predict_top_k(text: str, k: int = 5):
        if not text or model is None or tokenizer is None:
            return []
        try:
            import torch
            enc = tokenizer([text], truncation=True, padding=True, max_length=128, return_tensors="pt")
            with torch.no_grad():
                # Force CPU execution to avoid CUDA dependency in dashboard
                model.to("cpu")
                for k_inp, v_inp in enc.items():
                    enc[k_inp] = v_inp.to("cpu")
                logits = model(**enc).logits
                probs = torch.sigmoid(logits).cpu().numpy()[0]
            labels_list = classes
            if not labels_list:
                # fallback from config.json
                cfg = load_optional_json(os.path.join(model_dir, "config.json")) or {}
                id2label = cfg.get("id2label") or {}
                labels_list = [id2label.get(str(i), str(i)) for i in range(len(probs))]
            pairs = list(zip(labels_list, probs))
            pairs.sort(key=lambda x: x[1], reverse=True)
            return pairs[:k], probs
        except Exception as e:
            st.warning(f"Prediction failed: {e}")
            return []

    if st.button("Classify", type="primary"):
        topk, probs = predict_top_k(user_text)
        if topk:
            top_df = pd.DataFrame(topk, columns=["label", "probability"])  # top-5 bar chart
            fig_top = px.bar(top_df, x="label", y="probability", range_y=[0,1], title="Top-5 Predicted Labels")
            st.plotly_chart(fig_top, use_container_width=True)

            # Apply thresholds (if available) to show predicted categories
            if thresholds and isinstance(probs, np.ndarray):
                pred_labels = [label for label, p in zip(classes, probs) if p >= float(thresholds.get(label, 0.5))]
                st.markdown("Predicted categories (thresholded): " + ", ".join(pred_labels) if pred_labels else "None above threshold")
        else:
            st.info("Model not available or no input text provided.")
