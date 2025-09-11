import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import os
import json
import numpy as np
import re
import stopwordsiso as stopwords_iso

# Page configuration
st.set_page_config(page_title="AI Glow-rithms", layout="wide")

st.title("Dashboard")
st.markdown("Analyse the quality and relevance of comments through Share of Engagement (SoE)")

# Global white theme and UI polish
try:
    # Plotly defaults for a clean white theme with L'Oréal accents
    px.defaults.template = "plotly_white"
    px.defaults.color_discrete_sequence = [
        "#111111",  # Brand Black
        "#C8A97E",  # Brand Gold
        "#4B5563",  # Charcoal gray
        "#8B5E3C",  # Deep bronze
        "#9CA3AF",  # Neutral gray
        "#B08968"   # Soft bronze
    ]
    px.defaults.color_continuous_scale = [
        "#111111", "#3A2F20", "#6D5639", "#9B7B55", "#C8A97E"
    ]
except Exception:
    pass

# Inject CSS to enhance visuals (white background, cards, spacing, typography)
st.markdown(
    """
    <style>
      :root{
        --brand-black:#111111;
        --brand-gold:#C8A97E;
        --brand-charcoal:#4B5563;
        --border:#e5e7eb;
      }
      /* Base */
      .stApp { background-color: #ffffff; }
      .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
      }
      /* Typography */
      h1, h2, h3 { color: var(--brand-black); }
      h1 { font-weight: 700; letter-spacing: -0.01em; }
      h2 { font-weight: 600; }
      p, label, span, div { color: var(--brand-black); }
      /* Heading accent */
      h1:after{
        content: "";
        display: block;
        width: 64px; height: 3px;
        background: var(--brand-gold);
        margin-top: 8px;
        border-radius: 3px;
      }
      /* Tabs */
      div[role="tablist"] > div {
        background: #ffffff !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 10px rgba(17, 24, 39, 0.04);
      }
      button[role="tab"] { color: var(--brand-charcoal) !important; }
      button[role="tab"][data-baseweb="tab-highlighted"] {
        color: var(--brand-black) !important;
        box-shadow: inset 0 -3px 0 0 var(--brand-gold);
      }
      /* Metric cards */
      div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 14px 16px;
        box-shadow: 0 2px 12px rgba(17, 24, 39, 0.06);
        position: relative;
      }
      div[data-testid="stMetric"]:before{
        content: "";
        position: absolute; left: 0; top: 0; height: 4px; width: 100%;
        background: linear-gradient(90deg, var(--brand-gold), rgba(200,169,126,0.2));
        border-top-left-radius: 12px; border-top-right-radius: 12px;
      }
      /* Plotly charts */
      div[data-testid="stPlotlyChart"] {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 8px;
        box-shadow: 0 2px 12px rgba(17, 24, 39, 0.06);
      }
      /* Dataframes */
      div[data-testid="stDataFrame"] {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(17, 24, 39, 0.06);
      }
      /* Inputs & selects */
      .stTextInput, .stSelectbox, .stNumberInput, .stSlider { background: #ffffff; }
      /* Primary buttons */
      button[kind="primary"], .stButton>button {
        background: var(--brand-black) !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        border: 1px solid var(--brand-black) !important;
      }
      .stButton>button:hover {
        background: #000000 !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        border-color: var(--brand-gold) !important;
      }
      /* Subtle card utility for grouping */
      .card {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 12px rgba(17, 24, 39, 0.06);
        margin-bottom: 1rem;
        position: relative;
      }
      .card:before{
        content: "";
        position: absolute; left: 0; top: 0; height: 4px; width: 100%;
        background: var(--brand-gold);
        border-top-left-radius: 12px; border-top-right-radius: 12px;
      }
      /* Links */
      a { color: var(--brand-black); }
      a:hover { color: var(--brand-gold); }
    </style>
    """,
    unsafe_allow_html=True,
)

# Plotly brand styling helper
BRAND_BLACK = "#111111"
BRAND_GOLD = "#C8A97E"
BRAND_COLORWAY = ["#111111", "#C8A97E", "#4B5563", "#8B5E3C", "#9CA3AF", "#B08968"]

def apply_brand_style(fig):
    try:
        # Global layout
        fig.update_layout(
            colorway=BRAND_COLORWAY,
            title_font_color=BRAND_BLACK,
            font_color=BRAND_BLACK,
            legend_title_font_color=BRAND_BLACK,
            legend_font_color=BRAND_BLACK,
        )
        # If single trace, emphasize with brand gold
        if hasattr(fig, "data") and len(fig.data) <= 1:
            fig.update_traces(marker_color=BRAND_GOLD)
            fig.update_traces(marker_line_color=BRAND_BLACK, marker_line_width=0.5)
            fig.update_traces(line=dict(color=BRAND_BLACK, width=2.2))
    except Exception:
        pass

# Load data from selectable sources (auto-priority)
base_dir = os.path.dirname(__file__)
project_root = os.path.normpath(os.path.join(base_dir, ".."))
ai_root = os.path.join(project_root, "AI_Model")

# Candidate data sources
default_path = os.path.join(base_dir, "comments_with_sentiment.csv")
quality_path = os.path.join(ai_root, "QualityRelevanceSpamModel", "comments_evaluated.csv")
cluster_path = os.path.join(ai_root, "Clustering Model For Comment Sub Category", "clustered_comments_reassigned.csv")

# Load dataframe from the first available source
def _load_first_available_dataframe(paths):
    for path in paths:
        if os.path.exists(path) and os.path.isfile(path):
            try:
                return pd.read_csv(path)
            except Exception as e:
                st.warning(f"Failed to read {os.path.basename(path)}: {e}")
    return pd.DataFrame()

df = _load_first_available_dataframe([
    quality_path,
    cluster_path,
    default_path,
])

if df.empty:
    st.info("No data file found. Place a CSV at one of the expected paths to populate the dashboard.")

# Adding tab for better orgnasation
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Overview", "Sentiment", "Trends", "WordCloud", "AI Model Predictions", "Classification Model", "Spam Detection"])

# tab 1: Overview
with tab1:
    st.subheader("Key Metrics")
    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Comments", len(df))
    if "quality_score" in df.columns:
        col2.metric("Avg Quality Score", f"{df['quality_score'].mean():.2f}")
    if "relevance_score" in df.columns:
        col3.metric("Avg Relevance Score", f"{df['relevance_score'].mean():.2f}")

# tab 3: Trends
with tab3:
    st.subheader("Trends & Distributions")

    # 2. Quality distribution if present
    if "quality_score" in df.columns:
        st.subheader("Quality Score Distribution")
        # Server-side aggregation using numpy histogram
        try:
            counts, bin_edges = np.histogram(df["quality_score"].dropna().astype(float), bins=50)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            agg_df = pd.DataFrame({"bin": bin_centers, "count": counts})
            fig5 = px.bar(agg_df, x="bin", y="count", title="Distribution of Quality Scores")
            fig5.update_xaxes(title_text="quality_score")
            fig5.update_yaxes(title_text="count")
            apply_brand_style(fig5)
            st.plotly_chart(fig5, width='stretch', key="chart-quality-dist")
        except Exception:
            pass

    # 3. Sentiment distribution (already in tab2, but show again in trends)
    if "sentiment" in df.columns:
        st.subheader("Sentiment Distribution (Counts)")
        sent_series = df["sentiment"].value_counts().reset_index()
        sent_series.columns = ["sentiment", "count"]
        fig6 = px.bar(sent_series, x="sentiment", y="count",
                      title="Sentiment Counts")
        apply_brand_style(fig6)
        st.plotly_chart(fig6, width='stretch', key="chart-sentiment-counts")
        
        # Sentiment table with textOriginal
        if "textOriginal" in df.columns:
            st.subheader("Comments by Sentiment")
            
            # Filter controls
            col1, col2 = st.columns([2, 1])
            with col1:
                sentiment_filter_table = st.multiselect(
                    "Filter by sentiment:",
                    options=df["sentiment"].unique(),
                    default=df["sentiment"].unique(),
                    key="sentiment_table_filter"
                )
            with col2:
                sample_size_sentiment = st.number_input(
                    "Rows to show:", 
                    min_value=10, 
                    max_value=1000, 
                    value=100, 
                    step=10,
                    key="sentiment_sample_size"
                )
            
            # Filter and display data
            df_sentiment = df[df["sentiment"].isin(sentiment_filter_table)]
            display_cols = ["textOriginal", "sentiment"]
            
            # Add additional columns if available
            if "post_id" in df.columns:
                display_cols.insert(0, "post_id")
            if "comment_id" in df.columns:
                display_cols.insert(1, "comment_id")
            
            st.dataframe(
                df_sentiment[display_cols].head(int(sample_size_sentiment)), 
                width='stretch', 
                height=400
            )
            
            # Show sentiment counts for filtered data
            if len(sentiment_filter_table) > 0:
                filtered_counts = df_sentiment["sentiment"].value_counts()
                st.write("**Sentiment counts in filtered data:**")
                for sentiment, count in filtered_counts.items():
                    st.write(f"- {sentiment}: {count}")
        else:
            st.info("No 'textOriginal' column found to display comments with sentiment.")

    # 4. Category distribution (clusters/predictions)
    for col in ["new_cluster", "cluster", "predicted_category"]:
        if col in df.columns:
            st.subheader(f"Category Distribution — {col}")
            # Server-side aggregated counts
            cat_series = df[col].value_counts().reset_index()
            cat_series.columns = [col, "count"]
            fig7 = px.bar(cat_series.head(10), x=col, y="count",
                          title=f"Top 10 {col} categories")
            apply_brand_style(fig7)
            st.plotly_chart(fig7, width='stretch', key=f"chart-cat-{col}")

# tab 4: WordCloud
with tab4:
    st.subheader("Word Cloud of Comments")

    # Candidate text columns
    object_cols = [c for c in df.columns if df[c].dtype == "object"]
    preferred = [c for c in object_cols if c.lower() in {"textoriginal", "text", "comment", "comment_text", "caption", "title"}]
    default_col = preferred[0] if preferred else (object_cols[0] if object_cols else None)

    if default_col is None:
        st.info("No text columns available to build a word cloud.")
    else:
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            text_col = st.selectbox("Text column", options=object_cols, index=object_cols.index(default_col))
        with c2:
            max_words = st.slider("Max words", min_value=20, max_value=300, value=100, step=10)
        with c3:
            min_freq = st.slider("Min frequency", min_value=1, max_value=20, value=2, step=1)

        # Build stopwords set using stopwordsiso
        try:
            # Get English stopwords from stopwordsiso
            base_sw = set(stopwords_iso.stopwords("en"))
        except:
            # Fallback to WordCloud's built-in stopwords
            base_sw = set(WordCloud().stopwords)
        
        common_social = {"https", "http", "www", "com", "amp", "rt", "u", "ur", "im", "ya", "lol"}
        stopwords = base_sw.union(common_social)

        # Extract and clean text
        series = df[text_col].dropna().astype(str)
        # Remove URLs and mentions/hashtags markers but keep words
        series = series.str.replace(r"https?://\S+", " ", regex=True)
        series = series.str.replace(r"[@#]", " ", regex=True)
        
        # Process text in smaller batches to avoid MemoryError
        def batched_text_join(series, batch_size=10000):
            for start in range(0, len(series), batch_size):
                batch = series[start:start+batch_size]
                yield " \n ".join(batch.tolist()).lower()
        text_joined = " ".join(batched_text_join(series))

        # Tokenize to words (letters and apostrophes)
        tokens = re.findall(r"[a-zA-Z][a-zA-Z']+", text_joined)
        tokens = [t for t in tokens if len(t) > 1 and t not in stopwords]

        freq = Counter(tokens)
        # Apply min frequency filter
        if min_freq > 1:
            freq = Counter({k: v for k, v in freq.items() if v >= min_freq})

        if not freq:
            st.info("No tokens left after filtering; adjust min frequency or stopwords.")
        else:
            # Define keyword categories
            categories = {
                "Colors": ["red", "blue", "green", "yellow", "purple", "pink", "orange", "black", "white", "brown", "gray", "grey", "silver", "gold", "beige", "navy", "maroon", "teal", "coral", "mint", "lavender", "rose", "peach", "cream", "ivory", "tan", "olive", "turquoise", "magenta", "cyan", "lime", "indigo", "violet", "crimson", "scarlet", "burgundy", "emerald", "sapphire", "ruby", "amber", "copper", "bronze", "platinum"],
                "Textures": ["smooth", "rough", "soft", "hard", "silk", "velvet", "leather", "cotton", "wool", "satin", "matte", "glossy", "shiny", "dull", "fuzzy", "sleek", "textured", "bumpy", "gritty", "polished", "coarse", "fine", "thick", "thin", "dense", "lightweight", "heavy", "sturdy", "delicate", "robust"],
                "Feelings": ["love", "hate", "like", "dislike", "amazing", "awful", "beautiful", "ugly", "wonderful", "terrible", "fantastic", "horrible", "great", "bad", "good", "excellent", "poor", "perfect", "flawed", "incredible", "disappointing", "satisfied", "frustrated", "happy", "sad", "excited", "bored", "impressed", "disappointed", "pleased", "annoyed", "thrilled", "upset", "delighted", "furious", "content", "miserable", "ecstatic", "devastated"],
                "Quality": ["high", "low", "premium", "cheap", "expensive", "affordable", "luxury", "budget", "quality", "poor", "excellent", "terrible", "outstanding", "awful", "superior", "inferior", "top", "bottom", "best", "worst", "perfect", "flawed", "flawless", "defective", "durable", "fragile", "reliable", "unreliable", "consistent", "inconsistent", "professional", "amateur", "polished", "rough"],
                "Beauty": ["makeup", "skincare", "beauty", "cosmetic", "foundation", "concealer", "lipstick", "mascara", "eyeshadow", "blush", "bronzer", "highlighter", "primer", "moisturizer", "serum", "cleanser", "toner", "exfoliant", "mask", "cream", "lotion", "oil", "gel", "foam", "powder", "liquid", "stick", "pencil", "brush", "sponge", "applicator", "gloss", "matte", "shimmer", "sparkle", "glow", "radiant", "luminous", "dewy", "natural", "dramatic", "subtle", "bold", "soft", "intense"]
            }

            # Categorize keywords
            categorized_keywords = {}
            uncategorized = []
            
            for word, count in freq.items():
                categorized = False
                for category, keywords in categories.items():
                    if word in keywords:
                        if category not in categorized_keywords:
                            categorized_keywords[category] = {}
                        categorized_keywords[category][word] = count
                        categorized = True
                        break
                if not categorized:
                    uncategorized.append((word, count))

            # Display categorized keywords
            st.subheader("Keywords by Category")
            
            # Show category tabs
            if categorized_keywords:
                cat_tabs = st.tabs(list(categorized_keywords.keys()) + ["Other"])
                
                for i, (category, keywords) in enumerate(categorized_keywords.items()):
                    with cat_tabs[i]:
                        # Sort by frequency within category
                        sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
                        cat_df = pd.DataFrame(sorted_keywords, columns=["keyword", "count"])
                        cat_df["category"] = category
                        
                        # Calculate redundant keywords (words with same meaning/concept)
                        st.write(f"**{category} Keywords ({len(keywords)} unique words)**")
                        st.dataframe(cat_df, width='stretch', height=200)
                        
                        # Show total count for this category
                        total_count = sum(keywords.values())
                        st.metric(f"Total {category} mentions", total_count)
                        
                        # Show top 5 most frequent
                        if len(sorted_keywords) > 0:
                            st.write("**Top 5 most frequent:**")
                            for j, (word, count) in enumerate(sorted_keywords[:5]):
                                st.write(f"{j+1}. {word}: {count}")

                # Other/Uncategorized tab
                with cat_tabs[-1]:
                    if uncategorized:
                        other_df = pd.DataFrame(uncategorized, columns=["keyword", "count"])
                        other_df = other_df.sort_values("count", ascending=False)
                        st.write(f"**Other Keywords ({len(uncategorized)} words)**")
                        st.dataframe(other_df, width='stretch', height=200)
                    else:
                        st.write("No uncategorized keywords found.")

            # Show top keywords (original view)
            st.subheader("All Keywords (Top 50)")
            top_df = pd.DataFrame(freq.most_common(50), columns=["keyword", "count"])
            st.dataframe(top_df, width='stretch', height=240)

            # Generate word cloud from frequencies
            wc = WordCloud(width=1000, height=360, background_color="white")
            wc = wc.generate_from_frequencies(dict(freq.most_common(max_words)))
            plt.figure(figsize=(16, 6))
            plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

# tab 5: AI Model Predictions (Clusters table + accuracy)
with tab5:
    st.subheader("AI Model Predictions — Clusters Table")

    required_cols = {"textOriginal", "new_cluster"}
    if required_cols.issubset(df.columns):
        # Optional baseline for accuracy
        baseline_col = None
        if "cluster" in df.columns:
            baseline_col = "cluster"
        elif "original_cluster" in df.columns:
            baseline_col = "original_cluster"

        # Filters and display controls
        left, right = st.columns([2, 1])
        with left:
            selected_cluster = st.selectbox(
                "Filter by new_cluster (optional):",
                options=["(All)"] + sorted(df["new_cluster"].astype(str).unique().tolist()),
                index=0,
            )
        with right:
            sample_size = st.number_input("Rows to show", min_value=10, max_value=1000, value=100, step=10)

        df_view = df.copy()
        if selected_cluster != "(All)":
            df_view = df_view[df_view["new_cluster"].astype(str) == selected_cluster]

        display_cols = ["textOriginal", "new_cluster"]
        if baseline_col:
            display_cols.append(baseline_col)

        st.dataframe(df_view[display_cols].head(int(sample_size)), width='stretch', height=420)

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
        st.dataframe(df_show, width='stretch', height=320)
        try:
            fig_f1 = px.bar(df_show, x="label", y="f1-score", title="Top-10 Categories by Support — F1 Scores")
            apply_brand_style(fig_f1)
            st.plotly_chart(fig_f1, width='stretch', key="chart-cls-f1-top10")
        except Exception:
            pass
    elif os.path.exists(per_label_f1_path):
        per_label_df = pd.read_csv(per_label_f1_path)
        st.dataframe(per_label_df, width='stretch', height=320)
        try:
            fig_f1 = px.bar(per_label_df, x="label", y="f1", title="Per-Label F1 Scores")
            apply_brand_style(fig_f1)
            st.plotly_chart(fig_f1, width='stretch', key="chart-cls-f1-perlabel")
        except Exception:
            pass
    elif classes:
        st.info("No saved per-label metrics found. Provide classification_report.csv or per_label_f1.csv to visualize actual scores.")
        placeholder_df = pd.DataFrame({"label": classes, "f1": [None] * len(classes)})
        st.dataframe(placeholder_df, width='stretch', height=320)

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
            st.dataframe(comp_sorted, width='stretch', height=340)
            # Chart: show improvement deltas (top 10)
            fig_delta = px.bar(comp_sorted.head(10), x="label", y="delta", title="Top-10 Improvements after Threshold Tuning")
            apply_brand_style(fig_delta)
            st.plotly_chart(fig_delta, width='stretch', key="chart-cls-delta-tuning")
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
            apply_brand_style(fig_top)
            st.plotly_chart(fig_top, width='stretch', key="chart-cls-topk")

            # Apply thresholds (if available) to show predicted categories
            if thresholds and isinstance(probs, np.ndarray):
                pred_labels = [label for label, p in zip(classes, probs) if p >= float(thresholds.get(label, 0.5))]
                st.markdown("Predicted categories (thresholded): " + ", ".join(pred_labels) if pred_labels else "None above threshold")
        else:
            st.info("Model not available or no input text provided.")

# tab 7: Spam Model
with tab7:
    st.subheader("Spam Detection & Comment Relevancy Analysis")
    st.markdown("This module analyzes spam comments per video category and evaluates comment relevancy with video content.")

    # Check for required columns
    has_spam = "is_spam" in df.columns
    has_video_category = any(col in df.columns for col in ["video_category", "category", "video_type", "content_type"])
    has_text = "textOriginal" in df.columns
    has_quality = "quality_score" in df.columns
    has_relevance = "relevance_score" in df.columns

    if not has_spam:
        st.error("No 'is_spam' column found. Please ensure spam detection has been run on the dataset.")
        st.stop()

    # Overall spam distribution
    st.subheader("Overall Spam Distribution")
    spam_series = df["is_spam"].astype(str).str.lower().value_counts().reset_index()
    spam_series.columns = ["is_spam", "count"]
    
    col1, col2 = st.columns([2, 1])
    with col1:
        # Already server-side aggregated via value_counts
        fig_spam = px.bar(spam_series, x="is_spam", y="count", title="Spam vs Non-Spam Distribution")
        apply_brand_style(fig_spam)
        st.plotly_chart(fig_spam, width='stretch', key="chart-spam-overall")
    
    with col2:
        total_comments = len(df)
        spam_count = len(df[df["is_spam"].astype(str).str.lower() == "true"])
        spam_rate = (spam_count / total_comments) * 100 if total_comments > 0 else 0
        
        st.metric("Total Comments", total_comments)
        st.metric("Spam Comments", spam_count)
        st.metric("Spam Rate", f"{spam_rate:.1f}%")

    # Spam analysis per video category
    if has_video_category:
        st.subheader("Spam Analysis by Video Category")
        
        # Find the video category column
        video_cat_col = None
        for col in ["video_category", "category", "video_type", "content_type"]:
            if col in df.columns:
                video_cat_col = col
                break
        
        if video_cat_col:
            # Create spam analysis by category
            # Server-side groupby aggregation only
            spam_by_category = df.groupby([video_cat_col, "is_spam"]).size().unstack(fill_value=0)
            spam_by_category.columns = ["Non-Spam", "Spam"]
            spam_by_category["Total"] = spam_by_category["Non-Spam"] + spam_by_category["Spam"]
            spam_by_category["Spam_Rate"] = (spam_by_category["Spam"] / spam_by_category["Total"] * 100).round(1)
            spam_by_category = spam_by_category.sort_values("Spam_Rate", ascending=False)
            
            # Display table
            st.dataframe(spam_by_category, width='stretch')
            
            # Visualization
            agg_cat_df = spam_by_category.reset_index().copy()
            fig_category = px.bar(
                agg_cat_df, 
                x=video_cat_col, 
                y=["Non-Spam", "Spam"], 
                title="Spam Distribution by Video Category",
                barmode="stack"
            )
            apply_brand_style(fig_category)
            st.plotly_chart(fig_category, width='stretch', key="chart-spam-bycat")
            
            # Spam rate by category
            fig_rate = px.bar(
                agg_cat_df, 
                x=video_cat_col, 
                y="Spam_Rate", 
                title="Spam Rate by Video Category (%)"
            )
            apply_brand_style(fig_rate)
            st.plotly_chart(fig_rate, width='stretch', key="chart-spam-rate-bycat")

    # Comment relevancy analysis
    st.subheader("Comment Relevancy Analysis")
    
    if has_quality and has_relevance:
        # Quality vs Relevance scatter plot
        # Server-side 2D binning (hexbin-like via histogram2d)
        try:
            x = df["quality_score"].astype(float)
            y = df["relevance_score"].astype(float)
            H, xedges, yedges = np.histogram2d(x, y, bins=40)
            xcenters = (xedges[:-1] + xedges[1:]) / 2
            ycenters = (yedges[:-1] + yedges[1:]) / 2
            heat_df = pd.DataFrame([(xc, yc, int(H[i, j])) for i, xc in enumerate(xcenters) for j, yc in enumerate(ycenters)], columns=["quality", "relevance", "count"])
            heat_df = heat_df[heat_df["count"] > 0]
            fig_relevance = px.density_heatmap(heat_df, x="quality", y="relevance", z="count", nbinsx=40, nbinsy=40, title="Quality vs Relevancy (binned)")
            apply_brand_style(fig_relevance)
            st.plotly_chart(fig_relevance, width='stretch', key="chart-quality-vs-relevance-heat")
        except Exception:
            pass
        
        # Relevancy distribution
        # Server-side aggregated histogram for relevancy
        try:
            counts, bin_edges = np.histogram(df["relevance_score"].dropna().astype(float), bins=20)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            agg_rel = pd.DataFrame({"bin": bin_centers, "count": counts})
            fig_relevance_dist = px.bar(agg_rel, x="bin", y="count", title="Relevancy Score Distribution")
            fig_relevance_dist.update_xaxes(title_text="relevance_score")
            fig_relevance_dist.update_yaxes(title_text="count")
            apply_brand_style(fig_relevance_dist)
            st.plotly_chart(fig_relevance_dist, width='stretch', key="chart-relevance-dist")
        except Exception:
            pass
        
        
    elif has_quality:
        st.info("Only quality scores available. Relevancy scores not found in dataset.")
        # Quality distribution by spam status
        # Server-side aggregated histogram for quality by spam status
        try:
            tmp = df[["quality_score", "is_spam"]].dropna()
            tmp["quality_score"] = tmp["quality_score"].astype(float)
            # Compute separate histograms
            bins = np.linspace(tmp["quality_score"].min(), tmp["quality_score"].max(), 21)
            rows = []
            for label in sorted(tmp["is_spam"].astype(str).unique()):
                arr = tmp[tmp["is_spam"].astype(str) == label]["quality_score"].to_numpy()
                counts, edges = np.histogram(arr, bins=bins)
                centers = (edges[:-1] + edges[1:]) / 2
                for c, xval in zip(counts, centers):
                    rows.append({"bin": xval, "count": int(c), "is_spam": label})
            agg_quality = pd.DataFrame(rows)
            fig_quality = px.bar(agg_quality, x="bin", y="count", color="is_spam", barmode="group", title="Quality Score Distribution by Spam Status")
            fig_quality.update_xaxes(title_text="quality_score")
            fig_quality.update_yaxes(title_text="count")
            apply_brand_style(fig_quality)
            st.plotly_chart(fig_quality, width='stretch', key="chart-quality-dist-byspam")
        except Exception:
            pass
    else:
        st.info("No quality or relevancy scores found in dataset.")

    # Spam comments table with context
    if has_text:
        st.subheader("Spam Comments Analysis")
        
        # Filter controls
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            show_spam_only = st.checkbox("Show only spam comments", value=True)
        with col2:
            sample_size_spam = st.number_input("Rows to show", min_value=10, max_value=500, value=50, step=10)
        with col3:
            if has_video_category and video_cat_col:
                category_filter = st.selectbox("Filter by category", ["All"] + list(df[video_cat_col].unique()))
        
        # Apply filters
        df_filtered = df.copy()
        if show_spam_only:
            df_filtered = df_filtered[df_filtered["is_spam"].astype(str).str.lower() == "true"]
        if has_video_category and video_cat_col and category_filter != "All":
            df_filtered = df_filtered[df_filtered[video_cat_col] == category_filter]
        
        # Display columns
        display_cols = ["textOriginal", "is_spam"]
        if has_video_category and video_cat_col:
            display_cols.insert(1, video_cat_col)
        if has_quality:
            display_cols.append("quality_score")
        if has_relevance:
            display_cols.append("relevance_score")
        if "post_id" in df.columns:
            display_cols.insert(0, "post_id")
        
        st.dataframe(
            df_filtered[display_cols].head(int(sample_size_spam)), 
            width='stretch', 
            height=400
        )
        
        # Summary statistics
        if len(df_filtered) > 0:
            st.write("**Summary Statistics for Filtered Data:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Comments", len(df_filtered))
            with col2:
                if has_quality:
                    avg_quality = df_filtered["quality_score"].mean()
                    st.metric("Avg Quality Score", f"{avg_quality:.3f}")
            with col3:
                if has_relevance:
                    avg_relevance = df_filtered["relevance_score"].mean()
                    st.metric("Avg Relevancy Score", f"{avg_relevance:.3f}")
            with col4:
                spam_count_filtered = len(df_filtered[df_filtered["is_spam"].astype(str).str.lower() == "true"])
                st.metric("Spam Count", spam_count_filtered)
    else:
        st.info("No text column found to display spam comments.")
