import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# Use default matplotlib styling
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

# Use default Plotly settings


# Use default Plotly styling
def apply_brand_style(fig):
    # No custom styling - use Streamlit/Plotly defaults
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

filtered_df = df.copy()

# Adding tab for better orgnasation
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Overview", "Sentiment", "Trends", "WordCloud", "Cluster Analysis (Keywords)", "Classification Model", "Spam Detection"])

# tab 1: Overview
with tab1:
    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Comments", len(filtered_df))
    if "quality_score" in filtered_df.columns:
        col2.metric("Avg Quality Score", f"{filtered_df['quality_score'].mean():.2f}")
    if "relevance_score" in filtered_df.columns:
        col3.metric("Avg Relevance Score", f"{filtered_df['relevance_score'].mean():.2f}")
    st.divider()
    # Sentiment and Spam Pie Charts in the same row
    pie_col1, pie_col2 = st.columns(2)
    with pie_col1:
        if "sentiment" in filtered_df.columns:
            st.subheader("Sentiment Distribution (Pie Chart)")
            sent_counts = filtered_df["sentiment"].value_counts().reset_index()
            sent_counts.columns = ["sentiment", "count"]
            fig_sent = px.pie(sent_counts, names="sentiment", values="count", title="Sentiment Distribution")
            apply_brand_style(fig_sent)
            st.plotly_chart(fig_sent, use_container_width=True, key="overview-sentiment-pie")
    with pie_col2:
        if "is_spam" in filtered_df.columns:
            st.subheader("Spam vs Non-Spam (Pie Chart)")
            spam_counts = filtered_df["is_spam"].astype(str).str.lower().replace({"yes": "Spam", "no": "Non-Spam"}).value_counts().reset_index()
            spam_counts.columns = ["is_spam", "count"]
            fig_spam = px.pie(spam_counts, names="is_spam", values="count", title="Spam vs Non-Spam")
            apply_brand_style(fig_spam)
            st.plotly_chart(fig_spam, use_container_width=True, key="overview-spam-pie")
    st.divider()
    # Quality Score and Relevance Score Histograms in the same row
    hist_col1, hist_col2 = st.columns(2)
    with hist_col1:
        if "quality_score" in filtered_df.columns:
            st.subheader("Quality Score Distribution (Histogram)")
            fig_quality = px.histogram(filtered_df, x="quality_score", nbins=30, title="Quality Score Distribution")
            apply_brand_style(fig_quality)
            st.plotly_chart(fig_quality, use_container_width=True, key="overview-quality-hist")
    with hist_col2:
        if "relevance_score" in filtered_df.columns:
            st.subheader("Relevance Score Distribution (Histogram)")
            fig_relevance = px.histogram(filtered_df, x="relevance_score", nbins=30, title="Relevance Score Distribution")
            apply_brand_style(fig_relevance)
            st.plotly_chart(fig_relevance, use_container_width=True, key="overview-relevance-hist")
    st.divider()
    # Top 10 Clusters/Categories Bar Chart
    for col in ["new_cluster", "cluster", "predicted_category"]:
        if col in filtered_df.columns:
            st.subheader(f"Top 10 {col} Categories")
            cat_counts = filtered_df[col].value_counts().head(10).reset_index()
            cat_counts.columns = [col, "count"]
            fig_cat = px.bar(cat_counts, x=col, y="count", title=f"Top 10 {col} Categories")
            apply_brand_style(fig_cat)
            st.plotly_chart(fig_cat, width='stretch', key=f"overview-top10-{col}")

with tab2:
    if "sentiment" in filtered_df.columns:
        st.subheader("Sentiment Distribution (Counts)")
        sent_series = filtered_df["sentiment"].value_counts().reset_index()
        sent_series.columns = ["sentiment", "count"]
        fig6 = px.bar(sent_series, x="sentiment", y="count", title="Sentiment Counts")
        apply_brand_style(fig6)
        st.plotly_chart(fig6, width='stretch', key="chart-sentiment-counts")

        # Stacked bar: Sentiment distribution by cluster categories from cluster.txt
        try:
            cluster_file_path = os.path.join(os.path.dirname(__file__), "cluster.txt")
            if os.path.exists(cluster_file_path) and "textOriginal" in filtered_df.columns:
                with open(cluster_file_path, 'r', encoding='utf-8') as f:
                    lines = f.read().strip().split('\n')

                # Parse cluster.txt into a mapping: category -> [keywords]
                category_to_keywords = {}
                for line in lines:
                    if not line.strip() or ':' not in line:
                        continue
                    category_part, keywords_part = line.split(':', 1)
                    category_name = category_part.strip().strip("'\"")
                    kw_text = keywords_part.strip()
                    if kw_text.startswith('[') and kw_text.endswith(']'):
                        kw_text = kw_text[1:-1]
                    keywords = [kw.strip().strip("'\"") for kw in kw_text.split(',') if kw.strip()]
                    if keywords:
                        category_to_keywords[category_name] = keywords

                # Build sentiment counts per category using keyword presence in textOriginal
                if category_to_keywords:
                    text_series = filtered_df.get("textOriginal", pd.Series([], dtype=str)).fillna("").astype(str).str.lower()
                    records = []
                    for category_name, keywords in category_to_keywords.items():
                        # Build safe regex OR pattern of keywords
                        safe_parts = [re.escape(k.lower()) for k in keywords if isinstance(k, str) and k]
                        if not safe_parts:
                            continue
                        pattern = "|".join(safe_parts)
                        try:
                            mask = text_series.str.contains(pattern, regex=True, na=False)
                        except Exception:
                            # Fallback: skip problematic pattern
                            continue
                        if mask.any():
                            counts = filtered_df.loc[mask, "sentiment"].value_counts()
                            for sentiment_label, count in counts.items():
                                records.append({
                                    "Category": category_name,
                                    "sentiment": sentiment_label,
                                    "count": int(count)
                                })

                    if records:
                        agg_df = pd.DataFrame(records)
                        pivot_df = agg_df.pivot_table(index="Category", columns="sentiment", values="count", aggfunc="sum", fill_value=0)
                        # Keep only top 10 categories by total counts across sentiments
                        pivot_df["__total__"] = pivot_df.sum(axis=1)
                        pivot_df = pivot_df.sort_values("__total__", ascending=False).head(10)
                        pivot_df = pivot_df.drop(columns=["__total__"])  # remove helper column
                        stacked_df = pivot_df.reset_index()
                        y_cols = [c for c in stacked_df.columns if c not in {"Category"}]
                        fig_stack = px.bar(
                            stacked_df,
                            x="Category",
                            y=y_cols,
                            barmode="stack",
                            title="Sentiment Distribution by Cluster Category"
                        )
                        apply_brand_style(fig_stack)
                        st.plotly_chart(fig_stack, width='stretch', key="chart-sentiment-by-cluster")
                    else:
                        st.info("No keyword matches found in comments for cluster categories.")
            else:
                if "textOriginal" not in filtered_df.columns:
                    st.info("No 'textOriginal' column available to map keywords to comments.")
        except Exception as e:
            st.warning(f"Could not build stacked sentiment-by-cluster chart: {e}")
        if "textOriginal" in filtered_df.columns:
            st.subheader("Comments by Sentiment")
            col1, col2 = st.columns([2, 1])
            with col1:
                sentiment_filter_table = st.multiselect(
                    "Filter by sentiment:",
                    options=filtered_df["sentiment"].unique(),
                    default=filtered_df["sentiment"].unique(),
                    key="sentiment_table_filter"
                )
            with col2:
                sample_size_sentiment = st.number_input(
                    "Rows to show:", 
                    min_value=10, 
                    max_value=1000, 
                    value=10, 
                    step=5,
                    key="sentiment_sample_size"
                )
            df_sentiment = filtered_df[filtered_df["sentiment"].isin(sentiment_filter_table)]
            display_cols = ["textOriginal", "sentiment"]
            if "post_id" in filtered_df.columns:
                display_cols.insert(0, "post_id")
            if "comment_id" in filtered_df.columns:
                display_cols.insert(1, "comment_id")
            st.dataframe(
                df_sentiment[display_cols].head(int(float(sample_size_sentiment))).reset_index(drop=True).rename(
                    lambda x: x + 1, axis="index"
                ),
                height=400
            )
            if len(sentiment_filter_table) > 0:
                filtered_counts = df_sentiment["sentiment"].value_counts()
                st.write("**Sentiment counts in filtered data:**")
                # Show sentiment counts as metrics
                metric_cols = st.columns(len(filtered_counts))
                for i, (sentiment, count) in enumerate(filtered_counts.items()):
                    metric_cols[i].metric(f"{sentiment} Count", count)
        else:
            st.info("No 'textOriginal' column found to display comments with sentiment.")

# tab 3: Trends
with tab3:
    st.subheader("Trends & Distributions")
    dist_col1, dist_col2 = st.columns(2)
    with dist_col1:
        if "quality_score" in filtered_df.columns:
            st.subheader("Quality Score Distribution")
            try:
                counts, bin_edges = np.histogram(filtered_df["quality_score"].dropna().astype(float), bins=50)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                agg_df = pd.DataFrame({"bin": bin_centers, "count": counts})
                fig5 = px.bar(agg_df, x="bin", y="count", title="Distribution of Quality Scores")
                fig5.update_xaxes(title_text="quality_score")
                fig5.update_yaxes(title_text="count")
                apply_brand_style(fig5)
                st.plotly_chart(fig5, use_container_width=True, key="chart-quality-dist")
            except Exception:
                pass
    with dist_col2:
        if "relevance_score" in filtered_df.columns:
            st.subheader("Relevance Score Distribution")
            try:
                counts, bin_edges = np.histogram(filtered_df["relevance_score"].dropna().astype(float), bins=50)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                agg_df_rel = pd.DataFrame({"bin": bin_centers, "count": counts})
                fig_rel = px.bar(agg_df_rel, x="bin", y="count", title="Distribution of Relevance Scores")
                fig_rel.update_xaxes(title_text="relevance_score")
                fig_rel.update_yaxes(title_text="count")
                apply_brand_style(fig_rel)
                st.plotly_chart(fig_rel, use_container_width=True, key="chart-relevance-dist")
            except Exception:
                pass
    
    for col in ["new_cluster", "cluster", "predicted_category"]:
        if col in filtered_df.columns and "quality_score" in filtered_df.columns and "relevance_score" in filtered_df.columns:
            st.subheader(f"Average Quality & Relevance Score by {col}")
            cat_col1, cat_col2 = st.columns(2)
            with cat_col1:
                avg_quality = filtered_df.groupby(col)["quality_score"].mean().reset_index()
                avg_quality = avg_quality.sort_values("quality_score", ascending=False).head(10)
                fig_quality_cat = px.bar(
                    avg_quality,
                    x=col,
                    y="quality_score",
                    title=f"Top 10 {col} by Avg Quality Score"
                )
                apply_brand_style(fig_quality_cat)
                st.plotly_chart(fig_quality_cat, use_container_width=True, key=f"chart-quality-by-{col}")
            with cat_col2:
                avg_relevance = filtered_df.groupby(col)["relevance_score"].mean().reset_index()
                avg_relevance = avg_relevance.sort_values("relevance_score", ascending=False).head(10)
                fig_relevance_cat = px.bar(
                    avg_relevance,
                    x=col,
                    y="relevance_score",
                    title=f"Top 10 {col} by Avg Relevance Score"
                )
                apply_brand_style(fig_relevance_cat)
                st.plotly_chart(fig_relevance_cat, use_container_width=True, key=f"chart-relevance-by-{col}")

    for col in ["new_cluster", "cluster", "predicted_category"]:
        if col in filtered_df.columns:
            st.subheader(f"Category Distribution — {col}")
            cat_series = filtered_df[col].value_counts().reset_index()
            cat_series.columns = [col, "count"]
            fig7 = px.bar(cat_series.head(10), x=col, y="count", title=f"Top 10 {col} categories")
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
                        cat_df = cat_df.reset_index(drop=True)
                        cat_df.index = cat_df.index + 1
                        cat_df.index.name = "Index"
                        st.dataframe(cat_df, height=200)
                        
                        # Calculate redundant keywords (words with same meaning/concept)
                        st.write(f"**{category} Keywords ({len(keywords)} unique words)**")
                        st.dataframe(cat_df, height=200)
                        
                        # Show total count for this category
                        total_count = sum(keywords.values())
                        st.metric(f"Total {category} mentions", total_count)
                        
                        # Show top 5 most frequent
                        if len(sorted_keywords) > 0:
                            st.write("**Top 5 most frequent:**")
                            top5 = sorted_keywords[:5]
                            icons = ["😛", "😁", "🥳", "😉", "😊"] 
                            accent_colors = ["#C8A97E", "#B08968", "#8B5E3C", "#4B5563", "#9CA3AF"]
                            metric_cols = st.columns(5)
                            for idx, (word, count) in enumerate(top5):
                                icon = icons[idx] if idx < len(icons) else ""
                                color = accent_colors[idx] if idx < len(accent_colors) else "#111111"
                                metric_cols[idx].markdown(
                                    f"""
                                    <div style="text-align:center;">
                                        <span style="font-size:2.2em;">{icon}</span><br>
                                        <span style="font-size:1.1em; color:{color}; font-weight:600;">{word.title()}</span><br>
                                        <span style="font-size:1.3em; color:{color}; font-weight:700;">{count:,}</span>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                # Other/Uncategorized tab
                with cat_tabs[-1]:
                    if uncategorized:
                        other_df = pd.DataFrame(uncategorized, columns=["keyword", "count"])
                        other_df = other_df.sort_values("count", ascending=False)
                        other_df = other_df.reset_index(drop=True)
                        other_df.index = other_df.index + 1
                        other_df.index.name = "Index"
                        st.dataframe(other_df, height=200)
                    else:
                        st.write("No uncategorized keywords found.")

            # Show top keywords (original view)
            st.subheader("All Keywords (Top 50)")
            top_df = pd.DataFrame(freq.most_common(50), columns=["keyword", "count"])
            top_df = top_df.reset_index(drop=True)
            top_df.index = top_df.index + 1
            top_df.index.name = "Index"
            st.dataframe(top_df, height=240)

            # Generate word cloud from frequencies
            wc = WordCloud(width=1000, height=360, background_color="white")
            wc = wc.generate_from_frequencies(dict(freq.most_common(max_words)))
            plt.figure(figsize=(16, 6))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
    st.pyplot(plt)

# tab 5: Cluster Analysis (Keywords)
with tab5:
    st.subheader("Cluster Analysis — Keywords")
    st.markdown("Analysis of keywords extracted from comments and their assigned categories.")
    
    # Load cluster.txt file
    cluster_file_path = os.path.join(os.path.dirname(__file__), "cluster.txt")
    
    if os.path.exists(cluster_file_path):
        try:
            # Read the cluster.txt file
            with open(cluster_file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Parse the content to extract categories and keywords
            categories_data = []
            lines = content.strip().split('\n')
            
            for line in lines:
                if line.strip():
                    # Split by colon to separate category name from keywords
                    if ':' in line:
                        category_part, keywords_part = line.split(':', 1)
                        category_name = category_part.strip().strip("'\"")
                        
                        # Extract keywords from the list format
                        keywords_text = keywords_part.strip()
                        if keywords_text.startswith('[') and keywords_text.endswith(']'):
                            keywords_text = keywords_text[1:-1]  # Remove brackets
                        
                        # Split by comma and clean up keywords
                        keywords = [kw.strip().strip("'\"") for kw in keywords_text.split(',')]
                        
                        # Add each keyword with its category
                        for keyword in keywords:
                            if keyword:  # Skip empty keywords
                                categories_data.append({
                                    'Keyword': keyword,
                                    'Category': category_name
                                })
            
            if categories_data:
                # Create DataFrame
                keywords_df = pd.DataFrame(categories_data)
                
                # Count keyword frequencies
                keyword_counts = keywords_df['Keyword'].value_counts().reset_index()
                keyword_counts.columns = ['Keyword', 'Count']
                
                # Merge with categories
                result_df = keyword_counts.merge(
                    keywords_df[['Keyword', 'Category']].drop_duplicates(), 
                    on='Keyword', 
                    how='left'
                )
                
                # Sort by count (descending)
                result_df = result_df.sort_values('Count', ascending=False)
                
                # Display controls
                col1, col2 = st.columns([2, 1])
                with col1:
                    selected_category = st.selectbox(
                        "Filter by category:",
                        options=["(All)"] + sorted(result_df['Category'].unique().tolist()),
                        index=0,
                    )
                with col2:
                    max_keywords = st.number_input("Max keywords to show", min_value=10, max_value=500, value=100, step=10)
                
                # Filter by category if selected
                if selected_category != "(All)":
                    filtered_df = result_df[result_df['Category'] == selected_category]
                else:
                    filtered_df = result_df
                
                # Limit results
                display_df = filtered_df.head(max_keywords)
                display_df = display_df.reset_index(drop=True)
                display_df.index = display_df.index + 1
                display_df.index.name = "Index"
                st.dataframe(
                    display_df,
                    height=400,
                    use_container_width=True
                )
                
                # Display summary statistics
                st.subheader("Summary Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Keywords", len(result_df))
                with col2:
                    st.metric("Total Categories", result_df['Category'].nunique())
                with col3:
                    st.metric("Most Frequent Keyword", f"{result_df.iloc[0]['Keyword']} ({result_df.iloc[0]['Count']})")
                
                # Show top categories by keyword count
                st.subheader("Top Categories by Keyword Count")
                category_counts = result_df.groupby('Category')['Count'].sum().sort_values(ascending=False).head(10)
                category_df = pd.DataFrame({
                    'Category': category_counts.index,
                    'Total Keywords': category_counts.values
                })
                # Left-align the numbers by treating them as strings for display
                category_df['Total Keywords'] = category_df['Total Keywords'].astype(str)
                category_df = category_df.reset_index(drop=True)
                category_df.index = category_df.index + 1
                category_df.index.name = "Index"
                st.dataframe(category_df, height=300, use_container_width=True)
                
            else:
                st.warning("No keywords found in the cluster.txt file.")
                
        except Exception as e:
            st.error(f"Error reading cluster.txt file: {str(e)}")
    else:
        st.error("cluster.txt file not found. Please ensure the file exists in the Dashboard directory.")

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
        df_show = df_show.reset_index(drop=True)
        df_show.index = df_show.index + 1
        df_show.index.name = "Index"
        st.dataframe(df_show, height=320)
        try:
            fig_f1 = px.bar(df_show, x="label", y="f1-score", title="Top-10 Categories by Support — F1 Scores")
            apply_brand_style(fig_f1)
            st.plotly_chart(fig_f1, width='stretch', key="chart-cls-f1-top10")
        except Exception:
            pass
    elif os.path.exists(per_label_f1_path):
        per_label_df = pd.read_csv(per_label_f1_path)
        per_label_df = per_label_df.reset_index(drop=True)
        per_label_df.index = per_label_df.index + 1
        per_label_df.index.name = "Index"
        st.dataframe(per_label_df, height=320)
        try:
            fig_f1 = px.bar(per_label_df, x="label", y="f1", title="Per-Label F1 Scores")
            apply_brand_style(fig_f1)
            st.plotly_chart(fig_f1, width='stretch', key="chart-cls-f1-perlabel")
        except Exception:
            pass
    elif classes:
        placeholder_df = pd.DataFrame({"label": classes, "f1": [None] * len(classes)})
        placeholder_df = placeholder_df.reset_index(drop=True)
        placeholder_df.index = placeholder_df.index + 1
        placeholder_df.index.name = "Index"
        st.dataframe(placeholder_df, height=320)

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
            comp_sorted = comp_sorted.reset_index(drop=True)
            comp_sorted.index = comp_sorted.index + 1
            comp_sorted.index.name = "Index"
            st.dataframe(comp_sorted, height=340)
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

    col1, col2 = st.columns([2, 1])
    with col1:
        spam_counts = df["is_spam"].astype(str).str.lower().replace({"yes": "Spam", "no": "Non-Spam"}).value_counts().reset_index()
        spam_counts.columns = ["is_spam", "count"]
        spam_counts["label_with_count"] = spam_counts.apply(lambda row: f"{row['is_spam']} ({row['count']})", axis=1)
        fig_spam = px.pie(
            spam_counts,
            names="label_with_count",
            values="count",
            title="Spam vs Non-Spam"
        )
        apply_brand_style(fig_spam)
        st.plotly_chart(fig_spam, use_container_width=True, key="overview-tab7-pie")
            
    with col2:
        total_comments = len(df)
        spam_count = len(df[df["is_spam"].astype(str).str.lower() == "yes"])
        spam_rate = (spam_count / total_comments) * 100 if total_comments > 0 else 0
        st.metric("Total Comments", total_comments)
        st.metric("Spam Comments", spam_count)
        st.metric("Spam Rate", f"{spam_rate:.2f}%")

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
            spam_by_category = df.copy()
            spam_by_category["is_spam"] = spam_by_category["is_spam"].astype(str).str.lower().replace({"yes": "Spam", "no": "Non-Spam"})
            spam_by_category = spam_by_category.groupby([video_cat_col, "is_spam"]).size().unstack(fill_value=0)
            spam_by_category["Total"] = spam_by_category["Non-Spam"] + spam_by_category["Spam"]
            spam_by_category["Spam_Rate"] = (spam_by_category["Spam"] / spam_by_category["Total"] * 100).round(1)
            spam_by_category = spam_by_category.sort_values("Spam_Rate", ascending=False)
            
            # Display table
            st.dataframe(spam_by_category)
            
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
    
    # Spam analysis by topic category
    if "topicCategories_clean" in df.columns:
        st.subheader("Spam Rate Treemap by Topic Category")
        spam_topic_df = df.copy()
        spam_topic_df["is_spam"] = spam_topic_df["is_spam"].astype(str).str.lower().replace({"yes": "Spam", "no": "Non-Spam"})
        spam_topic_df["topicCategories_clean"] = spam_topic_df["topicCategories_clean"].apply(
            lambda x: re.findall(r"'([^']+)'", x) if isinstance(x, str) else []
        )
        spam_topic_exploded = spam_topic_df.explode("topicCategories_clean")
        agg = spam_topic_exploded.groupby("topicCategories_clean").agg(
            total_comments=("is_spam", "count"),
            spam_comments=("is_spam", lambda x: (x == "Spam").sum())
        )
        agg["Spam Rate (%)"] = (agg["spam_comments"] / agg["total_comments"] * 100).round(1)
        fig_treemap = px.treemap(
            agg.reset_index(),
            path=["topicCategories_clean"],
            values="total_comments",
            color="Spam Rate (%)",
            color_continuous_scale="Reds",
            title="Treemap: Spam Rate & Volume by Topic Category"
        )
        apply_brand_style(fig_treemap)
        st.plotly_chart(fig_treemap, use_container_width=True, key="chart-treemap-topiccat")

    # Spam comments table with context
    if has_text:
        st.subheader("Spam Comments Analysis")
        
        # Filter controls
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            all_topics = []
            if "topicCategories_clean" in df.columns:
                # Flatten all topics from all rows
                all_topics = sorted(set(
                    t for sublist in df["topicCategories_clean"].dropna().apply(
                        lambda x: re.findall(r"'([^']+)'", x) if isinstance(x, str) else []
                    ) for t in sublist
                ))
            selected_topics = st.multiselect(
                "Filter by topic category",
                options=all_topics,
                default=[],
            )
        with col2:
            sample_size_spam = st.number_input("Rows to show", min_value=10, max_value=500, value=10, step=5)
        with col3:
            if has_video_category and video_cat_col:
                category_filter = st.selectbox("Filter by category", ["All"] + list(df[video_cat_col].unique()))
        
        df_filtered = df.copy()
        # Filter by selected topic categories
        if selected_topics:
            df_filtered = df_filtered[
                df_filtered["topicCategories_clean"].apply(
                    lambda x: any(topic in x for topic in selected_topics) if isinstance(x, str) else False
                )
            ]
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
            df_filtered[display_cols].head(int(float(sample_size_spam))).reset_index(drop=True).rename(
                lambda x: x + 1, axis="index"
            ),
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
                spam_count_filtered = len(df_filtered[df_filtered["is_spam"].astype(str).str.lower() == "yes"])
                st.metric("Spam Count", spam_count_filtered)
    else:
        st.info("No text column found to display spam comments.")