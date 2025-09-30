"""
Additional Tab Modules for L'Oréal TrendSpotter Dashboard
Contains WordCloud, Cluster Analysis, and Spam Detection tabs
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re


def render_wordcloud_tab(filtered_df):
    """Render the WordCloud tab"""
    st.markdown('<h3 style="color:#000;">☁️ Word Cloud Analysis</h3>', unsafe_allow_html=True)
    
    # Find text columns with enhanced detection
    text_columns = [col for col in filtered_df.columns if any(keyword in col.lower() 
                   for keyword in ['comment', 'text', 'original'])]
    
    if not text_columns:
        st.warning("⚠️ No text columns found for word cloud generation.")
        st.info("💡 Available columns: " + ", ".join(filtered_df.columns.tolist()))
        return
    
    text_col = st.selectbox("📝 Select text column:", text_columns)
    
    # Word cloud options
    col1, col2 = st.columns(2)
    
    with col1:
        max_words = st.slider("📊 Maximum number of words:", 50, 500, 200)
    
    with col2:
        colormap = st.selectbox("🎨 Color scheme:", 
                               ["Reds", "Blues", "Greens", "Purples", "viridis", "plasma"])
    
    if st.button("🎨 Generate Word Cloud", type="primary"):
        try:
            with st.spinner("🔄 Generating word cloud..."):
                # Combine all text
                all_text = ' '.join(filtered_df[text_col].dropna().astype(str))
                
                if not all_text.strip():
                    st.warning("⚠️ No text data found to generate word cloud.")
                    return
                
                # Clean text
                all_text = clean_text_for_wordcloud(all_text)
                
                if not all_text.strip():
                    st.warning("⚠️ No valid words found after text cleaning.")
                    return
                
                # Generate word cloud
                wordcloud = WordCloud(
                    width=1200, 
                    height=600, 
                    background_color='white',
                    colormap=colormap,
                    max_words=max_words,
                    relative_scaling=0.5,
                    min_font_size=10
                ).generate(all_text)
                
                # Display word cloud
                fig, ax = plt.subplots(figsize=(15, 8))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                plt.tight_layout(pad=0)
                st.pyplot(fig)
                
                # Top words analysis
                render_word_frequency_analysis(all_text)
                
        except Exception as e:
            st.error(f"❌ Error generating word cloud: {str(e)}")


def clean_text_for_wordcloud(text):
    """Clean text for word cloud generation"""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove common stopwords
    stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
                'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
                'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
                'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their'}
    
    # Filter words
    words = [word for word in text.split() if len(word) > 2 and word not in stopwords]
    
    return ' '.join(words)


def render_word_frequency_analysis(text):
    """Render word frequency analysis"""
    
    st.markdown("### 📊 Word Frequency Analysis")
    
    # Calculate word frequencies
    word_freq = Counter(text.split())
    
    if not word_freq:
        st.warning("⚠️ No words found for frequency analysis.")
        return
    
    # Create tabs for different analyses
    freq_tab1, freq_tab2 = st.tabs(["📈 Top Words", "📊 Word Length Distribution"])
    
    with freq_tab1:
        # Top words chart
        top_n = st.slider("Number of top words to show:", 10, 50, 20)
        top_words = word_freq.most_common(top_n)
        
        words_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
        
        fig_words = px.bar(words_df, x='Word', y='Frequency', 
                          title=f'Top {top_n} Most Frequent Words')
        fig_words.update_xaxes(tickangle=45)
        st.plotly_chart(fig_words, use_container_width=True)
        
        # Display top words table
        st.dataframe(words_df, use_container_width=True)
    
    with freq_tab2:
        # Word length distribution
        word_lengths = [len(word) for word in word_freq.keys()]
        length_freq = Counter(word_lengths)
        
        length_df = pd.DataFrame(list(length_freq.items()), columns=['Length', 'Count'])
        length_df = length_df.sort_values('Length')
        
        fig_length = px.bar(length_df, x='Length', y='Count',
                           title='Word Length Distribution')
        st.plotly_chart(fig_length, use_container_width=True)


def render_cluster_tab(filtered_df, apply_brand_style):
    """Render the Cluster Analysis tab"""
    st.markdown('<h3 style="color:#000;">🏷️ Cluster Analysis (Keywords)</h3>', unsafe_allow_html=True)
    
    cluster_columns = [col for col in filtered_df.columns if 'cluster' in col.lower()]
    
    if not cluster_columns:
        st.warning("⚠️ No cluster columns found in the dataset.")
        st.info("💡 Available columns: " + ", ".join(filtered_df.columns.tolist()))
        return
    
    cluster_col = st.selectbox("🏷️ Select cluster column:", cluster_columns)
    
    # Cluster overview
    render_cluster_overview(filtered_df, cluster_col, apply_brand_style)
    
    # Cluster exploration
    render_cluster_exploration(filtered_df, cluster_col)


def render_cluster_overview(filtered_df, cluster_col, apply_brand_style):
    """Render cluster overview section"""
    
    st.markdown("### 📊 Cluster Distribution")
    
    cluster_counts = filtered_df[cluster_col].value_counts()
    
    # Display options
    col1, col2 = st.columns(2)
    
    with col1:
        show_top_n = st.slider("Show top N clusters:", 5, 25, 15)
    
    with col2:
        chart_type = st.selectbox("Chart type:", ["Bar Chart", "Pie Chart", "Both"])
    
    top_clusters = cluster_counts.head(show_top_n)
    
    if chart_type in ["Bar Chart", "Both"]:
        fig_bar = px.bar(x=top_clusters.index, y=top_clusters.values,
                        title=f'Top {show_top_n} {cluster_col} Distribution',
                        labels={'x': 'Cluster', 'y': 'Count'})
        fig_bar.update_xaxes(tickangle=45)
        apply_brand_style(fig_bar)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    if chart_type in ["Pie Chart", "Both"]:
        fig_pie = px.pie(values=top_clusters.values, names=top_clusters.index,
                        title=f'{cluster_col} Distribution (Top {show_top_n})')
        apply_brand_style(fig_pie)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Cluster statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🏷️ Total Clusters", f"{len(cluster_counts)}")
    
    with col2:
        st.metric("📊 Total Comments", f"{cluster_counts.sum():,}")
    
    with col3:
        avg_size = cluster_counts.mean()
        st.metric("📈 Avg Cluster Size", f"{avg_size:.1f}")
    
    with col4:
        largest_cluster = cluster_counts.iloc[0]
        st.metric("🔝 Largest Cluster", f"{largest_cluster:,}")


def render_cluster_exploration(filtered_df, cluster_col):
    """Render cluster exploration section"""
    
    st.markdown("### 🔍 Cluster Exploration")
    
    cluster_counts = filtered_df[cluster_col].value_counts()
    selected_cluster = st.selectbox("🎯 Select cluster to explore:", 
                                   cluster_counts.index.tolist())
    
    if selected_cluster:
        cluster_data = filtered_df[filtered_df[cluster_col] == selected_cluster]
        
        # Cluster information
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**📋 Cluster**: {selected_cluster}")
            st.info(f"**📊 Comments**: {len(cluster_data):,}")
        
        with col2:
            percentage = (len(cluster_data) / len(filtered_df)) * 100
            st.info(f"**📈 Percentage**: {percentage:.1f}%")
            
            if 'sentiment' in cluster_data.columns:
                sentiment_dist = cluster_data['sentiment'].value_counts()
                dominant_sentiment = sentiment_dist.index[0] if len(sentiment_dist) > 0 else 'N/A'
                st.info(f"**💭 Dominant Sentiment**: {dominant_sentiment}")
        
        # Sample comments from this cluster
        render_cluster_samples(cluster_data, selected_cluster)


def render_cluster_samples(cluster_data, cluster_name):
    """Render sample comments from selected cluster"""
    
    # Find text column
    text_columns = [col for col in cluster_data.columns if any(keyword in col.lower() 
                   for keyword in ['comment', 'text', 'original'])]
    
    if not text_columns:
        st.warning("⚠️ No text columns found to display sample comments.")
        return
    
    text_col = text_columns[0]  # Use first available text column
    
    st.markdown("#### 💬 Sample Comments")
    
    sample_size = st.slider("Number of sample comments:", 3, 20, 10, key=f"samples_{cluster_name}")
    
    valid_comments = cluster_data[cluster_data[text_col].notna() & (cluster_data[text_col].str.len() > 0)]
    
    if len(valid_comments) == 0:
        st.warning("⚠️ No valid comments found in this cluster.")
        return
    
    sample_comments = valid_comments.sample(min(sample_size, len(valid_comments)))
    
    for i, (idx, row) in enumerate(sample_comments.iterrows(), 1):
        comment_text = str(row[text_col])[:500] + "..." if len(str(row[text_col])) > 500 else str(row[text_col])
        
        with st.expander(f"💬 Comment {i}"):
            st.write(comment_text)
            
            # Show additional info if available
            additional_info = []
            if 'sentiment' in row.index and pd.notna(row['sentiment']):
                additional_info.append(f"Sentiment: {row['sentiment']}")
            if 'quality_score' in row.index and pd.notna(row['quality_score']):
                additional_info.append(f"Quality: {row['quality_score']:.2f}")
            if 'relevance_score' in row.index and pd.notna(row['relevance_score']):
                additional_info.append(f"Relevance: {row['relevance_score']:.2f}")
            
            if additional_info:
                st.caption(" | ".join(additional_info))


def render_spam_tab(filtered_df, apply_brand_style):
    """Render the Spam Detection tab"""
    st.markdown('<h3 style="color:#000;">🚫 Spam Detection Analysis</h3>', unsafe_allow_html=True)
    
    spam_columns = [col for col in filtered_df.columns if 'spam' in col.lower()]
    
    if not spam_columns:
        st.warning("⚠️ No spam detection columns found in the dataset.")
        st.info("💡 Available columns: " + ", ".join(filtered_df.columns.tolist()))
        return
    
    spam_col = st.selectbox("🚫 Select spam column:", spam_columns)
    
    # Spam overview
    render_spam_overview(filtered_df, spam_col, apply_brand_style)
    
    # Spam samples
    render_spam_samples(filtered_df, spam_col)


def render_spam_overview(filtered_df, spam_col, apply_brand_style):
    """Render spam detection overview"""
    
    st.markdown("### 📊 Spam Detection Overview")
    
    # Clean spam values
    spam_values = filtered_df[spam_col].astype(str).str.lower()
    spam_mapping = {
        'true': 'Spam', '1': 'Spam', 'yes': 'Spam', 'spam': 'Spam',
        'false': 'Non-Spam', '0': 'Non-Spam', 'no': 'Non-Spam', 'non-spam': 'Non-Spam'
    }
    
    spam_clean = spam_values.map(spam_mapping).fillna('Unknown')
    spam_counts = spam_clean.value_counts()
    
    total_comments = len(filtered_df)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    spam_count = spam_counts.get('Spam', 0)
    non_spam_count = spam_counts.get('Non-Spam', 0)
    
    with col1:
        spam_pct = (spam_count / total_comments * 100) if total_comments > 0 else 0
        st.metric("🚫 Spam Comments", f"{spam_count:,}", f"{spam_pct:.1f}%")
    
    with col2:
        non_spam_pct = (non_spam_count / total_comments * 100) if total_comments > 0 else 0
        st.metric("✅ Clean Comments", f"{non_spam_count:,}", f"{non_spam_pct:.1f}%")
    
    with col3:
        spam_ratio = non_spam_count / spam_count if spam_count > 0 else float('inf')
        ratio_text = f"1:{int(spam_ratio)}" if spam_ratio != float('inf') else "∞:1"
        st.metric("⚖️ Clean:Spam Ratio", ratio_text)
    
    with col4:
        unknown_count = spam_counts.get('Unknown', 0)
        st.metric("❓ Unknown Status", f"{unknown_count:,}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pie = px.pie(values=spam_counts.values, names=spam_counts.index,
                        title='Spam vs Clean Distribution')
        apply_brand_style(fig_pie)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        fig_bar = px.bar(x=spam_counts.index, y=spam_counts.values,
                        title='Spam Detection Results',
                        color=spam_counts.index,
                        color_discrete_map={'Spam': 'red', 'Non-Spam': 'green', 'Unknown': 'gray'})
        apply_brand_style(fig_bar)
        st.plotly_chart(fig_bar, use_container_width=True)


def render_spam_samples(filtered_df, spam_col):
    """Render spam sample comments"""
    
    st.markdown("### 💬 Sample Comments by Spam Status")
    
    # Clean spam values
    spam_values = filtered_df[spam_col].astype(str).str.lower()
    spam_mapping = {
        'true': 'Spam', '1': 'Spam', 'yes': 'Spam', 'spam': 'Spam',
        'false': 'Non-Spam', '0': 'Non-Spam', 'no': 'Non-Spam', 'non-spam': 'Non-Spam'
    }
    
    filtered_df = filtered_df.copy()
    filtered_df['spam_status'] = spam_values.map(spam_mapping).fillna('Unknown')
    
    spam_filter = st.selectbox("🔍 Filter by spam status:", 
                              ["All", "Spam", "Non-Spam", "Unknown"])
    
    if spam_filter != "All":
        sample_df = filtered_df[filtered_df['spam_status'] == spam_filter]
    else:
        sample_df = filtered_df
    
    if len(sample_df) == 0:
        st.warning(f"⚠️ No comments found with status: {spam_filter}")
        return
    
    # Find text column
    text_columns = [col for col in sample_df.columns if any(keyword in col.lower() 
                   for keyword in ['comment', 'text', 'original'])]
    
    if not text_columns:
        st.warning("⚠️ No text columns found to display comments.")
        return
    
    text_col = text_columns[0]
    sample_size = min(10, len(sample_df))
    
    if sample_size > 0:
        sample_comments = sample_df.sample(sample_size)
        
        for idx, row in sample_comments.iterrows():
            status = row['spam_status']
            status_emoji = {"Spam": "🚫", "Non-Spam": "✅", "Unknown": "❓"}.get(status, "❓")
            
            comment_text = str(row[text_col])
            preview_text = comment_text[:100] + "..." if len(comment_text) > 100 else comment_text
            
            with st.expander(f"{status_emoji} {status} - {preview_text}"):
                st.write(f"**Full Comment:** {comment_text}")
                
                # Additional info
                additional_info = []
                if 'quality_score' in row.index and pd.notna(row['quality_score']):
                    additional_info.append(f"Quality: {row['quality_score']:.2f}")
                if 'relevance_score' in row.index and pd.notna(row['relevance_score']):
                    additional_info.append(f"Relevance: {row['relevance_score']:.2f}")
                if 'sentiment' in row.index and pd.notna(row['sentiment']):
                    additional_info.append(f"Sentiment: {row['sentiment']}")
                
                if additional_info:
                    st.caption(" | ".join(additional_info))