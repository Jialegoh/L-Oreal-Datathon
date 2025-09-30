"""
Sentiment Analysis Tab Module for L'Oréal TrendSpotter Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px


def render_sentiment_tab(filtered_df, apply_brand_style):
    """
    Render the Sentiment Analysis tab
    
    Args:
        filtered_df (pd.DataFrame): Filtered dataset
        apply_brand_style (function): Function to apply L'Oréal brand styling to plots
    """
    st.markdown('<h3 style="color:#000;">Sentiment Analysis</h3>', unsafe_allow_html=True)
    
    if "sentiment" not in filtered_df.columns:
        st.error("❌ Sentiment column not found in the dataset.")
        return
    
    # Sentiment overview metrics
    col1, col2, col3 = st.columns(3)
    
    sentiment_counts = filtered_df["sentiment"].value_counts()
    total_comments = len(filtered_df)
    
    # Handle different possible sentiment value formats
    def get_sentiment_count(sentiment_type):
        """Get count for sentiment type, handling various formats"""
        count = 0
        sentiment_lower = filtered_df["sentiment"].astype(str).str.lower()
        
        if sentiment_type == "positive":
            # Look for positive variations
            positive_terms = ["positive", "pos", "1", "good", "happy", "joy"]
            count = sentiment_lower.isin(positive_terms).sum()
        elif sentiment_type == "negative":
            # Look for negative variations
            negative_terms = ["negative", "neg", "0", "bad", "sad", "anger"]
            count = sentiment_lower.isin(negative_terms).sum()
        elif sentiment_type == "neutral":
            # Look for neutral variations
            neutral_terms = ["neutral", "neu", "2", "mixed", "none"]
            count = sentiment_lower.isin(neutral_terms).sum()
        
        return count
    
    with col1:
        positive_count = get_sentiment_count("positive")
        positive_pct = (positive_count / total_comments * 100) if total_comments > 0 else 0
        st.metric("Positive Comments", f"{positive_count:,}", f"{positive_pct:.1f}%")
    
    with col2:
        negative_count = get_sentiment_count("negative")
        negative_pct = (negative_count / total_comments * 100) if total_comments > 0 else 0
        st.metric("Negative Comments", f"{negative_count:,}", f"{negative_pct:.1f}%")
    
    with col3:
        neutral_count = get_sentiment_count("neutral")
        neutral_pct = (neutral_count / total_comments * 100) if total_comments > 0 else 0
        st.metric("Neutral Comments", f"{neutral_count:,}", f"{neutral_pct:.1f}%")
    
    st.divider()
    
    # Sentiment distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        sent_counts = filtered_df["sentiment"].value_counts().reset_index()
        sent_counts.columns = ["sentiment", "count"]
        fig_pie = px.pie(sent_counts, names="sentiment", values="count", 
                        title="Sentiment Distribution (Pie Chart)")
        apply_brand_style(fig_pie)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Bar chart
        fig_bar = px.bar(sent_counts, x="sentiment", y="count", 
                        title="Sentiment Distribution (Bar Chart)")
        apply_brand_style(fig_bar)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Sentiment over time (if date column exists)
    if any(col for col in filtered_df.columns if 'date' in col.lower()):
        date_col = next(col for col in filtered_df.columns if 'date' in col.lower())
        st.markdown('<h4 style="color:#000;">Sentiment Trends Over Time</h4>', unsafe_allow_html=True)
        
        try:
            df_time = filtered_df.copy()
            df_time[date_col] = pd.to_datetime(df_time[date_col])
            df_time['date'] = df_time[date_col].dt.date
            
            sentiment_time = df_time.groupby(['date', 'sentiment']).size().reset_index(name='count')
            
            fig_time = px.line(sentiment_time, x='date', y='count', color='sentiment',
                             title='Sentiment Trends Over Time')
            apply_brand_style(fig_time)
            st.plotly_chart(fig_time, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not create time series chart: {str(e)}")
    
    # Sample comments by sentiment
    st.markdown('<h4 style="color:#000;">Sample Comments by Sentiment</h4>', unsafe_allow_html=True)
    
    sentiment_filter = st.selectbox("Select Sentiment to View Sample Comments:", 
                                   options=["All"] + list(sentiment_counts.index))
    
    try:
        if sentiment_filter == "All":
            if len(filtered_df) > 0:
                sample_df = filtered_df.sample(min(10, len(filtered_df)), random_state=42)
            else:
                st.warning("⚠️ No comments available to display")
                return
        else:
            # Filter by specific sentiment
            sentiment_filtered = filtered_df[filtered_df["sentiment"] == sentiment_filter]
            
            if len(sentiment_filtered) > 0:
                sample_df = sentiment_filtered.sample(min(10, len(sentiment_filtered)), random_state=42)
            else:
                st.warning(f"⚠️ No comments found with sentiment '{sentiment_filter}'")
                return
        
        # Determine the comment column to use
        comment_col = None
        if "comment" in sample_df.columns:
            comment_col = "comment"
        elif "text" in sample_df.columns:
            comment_col = "text"
        elif "textOriginal" in sample_df.columns:
            comment_col = "textOriginal"
        else:
            st.error("❌ No comment, text, or textOriginal column found in the dataset")
            return
        
        # Display comments in a table format
        st.markdown("### 📝 Sample Comments Table")
        
        # Create a display dataframe with better error handling
        try:
            display_df = sample_df[["sentiment", comment_col]].copy()
            
            # Handle null values
            display_df = display_df.dropna()
            
            if display_df.empty:
                st.warning("⚠️ No valid data to display after removing null values")
                return
            
            # Rename columns for display
            display_df.columns = ["Sentiment", "Comment"]
            display_df.reset_index(drop=True, inplace=True)
            
            # Truncate long comments for table display
            display_df["Comment"] = display_df["Comment"].astype(str).apply(
                lambda x: x[:200] + "..." if len(str(x)) > 200 else str(x)
            )
            
            # Display the sample comments table with modern styling
            # Reset index to start from 1
            display_df.index = range(1, len(display_df) + 1)
            
            # Apply clean minimalist styling with CSS (force override existing styles)
            st.markdown("""
            <style>
            .clean-table {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif !important;
                border-collapse: collapse !important;
                width: 100% !important;
                margin: 20px 0 !important;
                background: #ffffff !important;
                border-radius: 8px !important;
                overflow: hidden !important;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
            }
            .clean-table th {
                background-color: #000000 !important;
                background: #000000 !important;
                background-image: none !important;
                color: #ffffff !important;
                font-weight: 600 !important;
                text-align: left !important;
                padding: 16px 20px !important;
                font-size: 12px !important;
                text-transform: uppercase !important;
                letter-spacing: 0.05em !important;
                border: none !important;
            }
            .clean-table thead th {
                background-color: #000000 !important;
                background: #000000 !important;
                background-image: none !important;
                color: #ffffff !important;
            }
            .clean-table th:not(:hover) {
                background-color: #000000 !important;
                background: #000000 !important;
                background-image: none !important;
                color: #ffffff !important;
            }
            .clean-table thead th:not(:hover) {
                background-color: #000000 !important;
                background: #000000 !important;
                background-image: none !important;
                color: #ffffff !important;
            }
            table.clean-table th {
                background-color: #000000 !important;
                background: #000000 !important;
                background-image: none !important;
                color: #ffffff !important;
            }
            table.clean-table thead th {
                background-color: #000000 !important;
                background: #000000 !important;
                background-image: none !important;
                color: #ffffff !important;
            }
            .clean-table th:hover {
                background-color: #000000 !important;
                background: #000000 !important;
                background-image: none !important;
                color: #ffffff !important;
            }
            .clean-table thead th:hover {
                background-color: #000000 !important;
                background: #000000 !important;
                background-image: none !important;
                color: #ffffff !important;
            }
            table.clean-table th:hover {
                background-color: #000000 !important;
                background: #000000 !important;
                background-image: none !important;
                color: #ffffff !important;
            }
            table.clean-table thead th:hover {
                background-color: #000000 !important;
                background: #000000 !important;
                background-image: none !important;
                color: #ffffff !important;
            }
            .clean-table td {
                padding: 16px 20px !important;
                border-bottom: 1px solid #e5e7eb !important;
                vertical-align: top !important;
                font-size: 14px !important;
                line-height: 1.5 !important;
                color: #111827 !important;
                background-color: #ffffff !important;
                background: #ffffff !important;
            }
            .clean-table tbody td {
                background-color: #ffffff !important;
                background: #ffffff !important;
                color: #111827 !important;
            }
            .clean-table tr:last-child td {
                border-bottom: none !important;
            }
            .clean-table tbody tr:hover td {
                background-color: #f3f4f6 !important;
                background: #f3f4f6 !important;
                color: #111827 !important;
            }
            .clean-table thead tr:hover th {
                background-color: #000000 !important;
                background: #000000 !important;
                background-image: none !important;
                color: #ffffff !important;
            }
            .clean-table tr:hover th {
                background-color: #000000 !important;
                background: #000000 !important;
                background-image: none !important;
                color: #ffffff !important;
            }
            .clean-table .index-col {
                font-weight: 600 !important;
                color: #6b7280 !important;
                text-align: center !important;
                width: 60px !important;
                font-size: 13px !important;
                background-color: #ffffff !important;
                background: #ffffff !important;
            }
            .clean-table .sentiment-col {
                font-weight: 500 !important;
                width: 120px !important;
                color: #374151 !important;
                text-transform: capitalize !important;
                background-color: #ffffff !important;
                background: #ffffff !important;
            }
            .clean-table .comment-col {
                color: #111827 !important;
                line-height: 1.6 !important;
                max-width: 600px !important;
                background-color: #ffffff !important;
                background: #ffffff !important;
            }
            .clean-table tbody tr:hover .index-col {
                background-color: #f3f4f6 !important;
                background: #f3f4f6 !important;
                color: #6b7280 !important;
            }
            .clean-table tbody tr:hover .sentiment-col {
                background-color: #f3f4f6 !important;
                background: #f3f4f6 !important;
                color: #374151 !important;
            }
            .clean-table tbody tr:hover .comment-col {
                background-color: #f3f4f6 !important;
                background: #f3f4f6 !important;
                color: #111827 !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Create HTML table with clean minimalist styling
            html_table = '<table class="clean-table">'
            html_table += '<thead><tr>'
            html_table += '<th class="index-col">#</th>'
            html_table += '<th class="sentiment-col">Sentiment</th>'
            html_table += '<th class="comment-col">Comment</th>'
            html_table += '</tr></thead><tbody>'
            
            for idx, row in display_df.iterrows():
                # Clean sentiment display without emojis
                sentiment = row['Sentiment'].title()
                
                html_table += f'<tr>'
                html_table += f'<td class="index-col">{idx}</td>'
                html_table += f'<td class="sentiment-col">{sentiment}</td>'
                html_table += f'<td class="comment-col">{row["Comment"]}</td>'
                html_table += f'</tr>'
            
            html_table += '</tbody></table>'
            
            st.markdown(html_table, unsafe_allow_html=True)
                
        except Exception as table_error:
            st.error(f"❌ Error creating display table: {str(table_error)}")
            
            # Fallback: show raw data
            st.markdown("### 🔧 Raw Data (Fallback)")
            st.write("Sample DataFrame content:")
            st.write(sample_df[["sentiment", comment_col]].head())
        

    except Exception as e:
        st.error(f"❌ Error displaying sample comments: {str(e)}")
        st.info("🔍 Debug info:")
        st.write(f"- Dataset shape: {filtered_df.shape}")
        st.write(f"- Sentiment filter: {sentiment_filter}")
        st.write(f"- Available columns: {list(filtered_df.columns)}")
        if len(filtered_df) > 0:
            st.write(f"- Sample of sentiment values: {filtered_df['sentiment'].head().tolist()}")