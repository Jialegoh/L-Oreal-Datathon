"""
Additional Tab Modules for L'Oréal TrendSpotter Dashboard
Contains Trends, WordCloud, Cluster Analysis, and Spam Detection tabs
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re


def render_trends_tab(filtered_df, apply_brand_style):
    """Render the Trends Analysis tab"""
    st.markdown('<h3 style="color:#000;">Trends Analysis</h3>', unsafe_allow_html=True)
    
    # Time-based analysis if date column exists
    date_columns = [col for col in filtered_df.columns if 'date' in col.lower()]
    
    if not date_columns:
        st.warning("⚠️ No date columns found for trend analysis.")
        return
    
    date_col = st.selectbox("Select date column:", date_columns)
    
    try:
        df_trends = filtered_df.copy()
        df_trends[date_col] = pd.to_datetime(df_trends[date_col])
        df_trends['date'] = df_trends[date_col].dt.date
        
        # Daily trends
        daily_counts = df_trends.groupby('date').size().reset_index(name='count')
        fig_daily = px.line(daily_counts, x='date', y='count', title='Daily Comment Volume')
        apply_brand_style(fig_daily)
        st.plotly_chart(fig_daily, use_container_width=True)
        
        # Sentiment trends if available
        if 'sentiment' in filtered_df.columns:
            sentiment_trends = df_trends.groupby(['date', 'sentiment']).size().reset_index(name='count')
            fig_sentiment = px.line(sentiment_trends, x='date', y='count', color='sentiment',
                                  title='Sentiment Trends Over Time')
            apply_brand_style(fig_sentiment)
            st.plotly_chart(fig_sentiment, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error creating trends: {str(e)}")


def render_wordcloud_tab(filtered_df):
    """Render the WordCloud tab"""
    st.markdown('<h3 style="color:#000;">Word Cloud Analysis</h3>', unsafe_allow_html=True)
    
    text_columns = [col for col in filtered_df.columns if 'comment' in col.lower() or 'text' in col.lower()]
    
    if not text_columns:
        st.warning("⚠️ No text columns found for word cloud generation.")
        return
    
    text_col = st.selectbox("Select text column:", text_columns)
    
    # Word cloud options
    max_words = st.slider("Maximum number of words:", 50, 500, 200)
    
    if st.button("🎨 Generate Word Cloud"):
        try:
            # Combine all text
            all_text = ' '.join(filtered_df[text_col].dropna().astype(str))
            
            # Clean text
            all_text = re.sub(r'[^a-zA-Z\s]', '', all_text)
            all_text = ' '.join([word for word in all_text.split() if len(word) > 2])
            
            # Generate word cloud
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                colormap='Reds',
                max_words=max_words
            ).generate(all_text)
            
            # Display
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            
            # Top words
            st.markdown('<h4 style="color:#000;">Top Words</h4>', unsafe_allow_html=True)
            word_freq = Counter(all_text.split())
            top_words = word_freq.most_common(20)
            
            words_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
            fig_words = px.bar(words_df, x='Word', y='Frequency', title='Top 20 Most Frequent Words')
            st.plotly_chart(fig_words, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error generating word cloud: {str(e)}")


def render_cluster_tab(filtered_df, apply_brand_style):
    """Render the Cluster Analysis tab"""
    st.markdown('<h3 style="color:#000;">Cluster Analysis (Keywords)</h3>', unsafe_allow_html=True)
    
    cluster_columns = [col for col in filtered_df.columns if 'cluster' in col.lower()]
    
    if not cluster_columns:
        st.warning("⚠️ No cluster columns found in the dataset.")
        return
    
    cluster_col = st.selectbox("Select cluster column:", cluster_columns)
    
    # Cluster distribution
    cluster_counts = filtered_df[cluster_col].value_counts().head(15)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_bar = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                        title=f'Top 15 {cluster_col} Distribution')
        apply_brand_style(fig_bar)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        fig_pie = px.pie(values=cluster_counts.values, names=cluster_counts.index,
                        title=f'{cluster_col} Distribution (Top 15)')
        apply_brand_style(fig_pie)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Cluster details
    st.markdown('<h4 style="color:#000;">Cluster Details</h4>', unsafe_allow_html=True)
    
    selected_cluster = st.selectbox("Select cluster to explore:", cluster_counts.index)
    
    if selected_cluster:
        cluster_data = filtered_df[filtered_df[cluster_col] == selected_cluster]
        st.write(f"**Cluster**: {selected_cluster}")
        st.write(f"**Number of comments**: {len(cluster_data)}")
        
        # Sample comments from this cluster
        if 'comment' in cluster_data.columns:
            sample_comments = cluster_data['comment'].dropna().head(5)
            st.write("**Sample comments:**")
            for i, comment in enumerate(sample_comments, 1):
                st.write(f"{i}. {comment}")


def render_spam_tab(filtered_df, apply_brand_style):
    """Render the Spam Detection tab"""
    st.markdown('<h3 style="color:#000;">Spam Detection Analysis</h3>', unsafe_allow_html=True)
    
    if 'is_spam' not in filtered_df.columns:
        st.warning("⚠️ No spam detection column ('is_spam') found in the dataset.")
        return
    
    # Spam statistics
    spam_counts = filtered_df['is_spam'].value_counts()
    total_comments = len(filtered_df)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        spam_count = spam_counts.get('yes', 0) + spam_counts.get(True, 0) + spam_counts.get('Spam', 0)
        spam_pct = (spam_count / total_comments * 100) if total_comments > 0 else 0
        st.metric("Spam Comments", f"{spam_count:,}", f"{spam_pct:.1f}%")
    
    with col2:
        not_spam_count = spam_counts.get('no', 0) + spam_counts.get(False, 0) + spam_counts.get('Non-Spam', 0)
        not_spam_pct = (not_spam_count / total_comments * 100) if total_comments > 0 else 0
        st.metric("Non-Spam Comments", f"{not_spam_count:,}", f"{not_spam_pct:.1f}%")
    
    with col3:
        spam_ratio = spam_count / not_spam_count if not_spam_count > 0 else 0
        st.metric("Spam Ratio", f"1:{int(1/spam_ratio) if spam_ratio > 0 else 0}")
    
    # Spam distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Clean up spam labels for display
        spam_display = filtered_df['is_spam'].astype(str).str.lower()
        spam_display = spam_display.replace({'yes': 'Spam', 'no': 'Non-Spam', 'true': 'Spam', 'false': 'Non-Spam'})
        spam_counts_clean = spam_display.value_counts()
        
        fig_pie = px.pie(values=spam_counts_clean.values, names=spam_counts_clean.index,
                        title='Spam vs Non-Spam Distribution')
        apply_brand_style(fig_pie)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        fig_bar = px.bar(x=spam_counts_clean.index, y=spam_counts_clean.values,
                        title='Spam Detection Results')
        apply_brand_style(fig_bar)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Sample spam comments
    st.markdown('<h4 style="color:#000;">Sample Comments</h4>', unsafe_allow_html=True)
    
    spam_filter = st.selectbox("Filter by spam status:", ["All", "Spam", "Non-Spam"])
    
    if spam_filter != "All":
        is_spam_value = 'yes' if spam_filter == "Spam" else 'no'
        sample_df = filtered_df[filtered_df['is_spam'].astype(str).str.lower() == is_spam_value]
    else:
        sample_df = filtered_df
    
    sample_size = min(10, len(sample_df))
    if sample_size > 0:
        sample_comments = sample_df.sample(sample_size)
        
        if 'comment' in sample_comments.columns:
            for idx, row in sample_comments.iterrows():
                spam_status = "🚫 SPAM" if str(row['is_spam']).lower() in ['yes', 'true'] else "✅ CLEAN"
                with st.expander(f"{spam_status} - Comment"):
                    st.write(row['comment'])