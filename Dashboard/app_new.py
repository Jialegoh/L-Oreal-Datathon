"""
L'Oréal TrendSpotter Dashboard - Refactored Main Application
Professional Streamlit dashboard for AI-powered social media analytics

This refactored version imports functionality from separate modules:
- styles.py: L'Oréal branded CSS styling
- utils.py: Utility functions for data loading and UI components
- model_utils.py: AI model operations and GPU acceleration
- tab_*.py: Individual tab implementations
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json

# Import custom modules
from styles import get_loreal_styles
from utils import (
    encode_logo_to_base64, 
    create_header_html, 
    load_data_with_fallback, 
    get_data_paths,
    display_device_info
)
from model_utils import load_model_bundle, get_device
from tab_overview import render_overview_tab
from tab_sentiment import render_sentiment_tab
from tab_classification import render_classification_tab
from tab_trends import render_trends_tab
from tab_additional import (
    render_wordcloud_tab, 
    render_cluster_tab, 
    render_spam_tab
)


# ================================
# PAGE CONFIGURATION
# ================================
st.set_page_config(
    page_title="TrendSpotter - L'Oréal AI Analytics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ================================
# STYLING AND BRANDING
# ================================
def apply_brand_style(fig):
    """Apply L'Oréal brand styling to Plotly figures"""
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color='#000000',
        title_font_color='#000000',
        title_font_size=16,
        font_size=12,
        font_family="Arial, sans-serif",
        colorway=['#ED1B2E', '#000000', '#6D6E70', '#D7D7D8', '#FF6B7A', '#8B0000']
    )
    
    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#E5E7EB',
        zeroline=False,
        title_font_color='#000000',
        tickfont_color='#000000'
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#E5E7EB',
        zeroline=False,
        title_font_color='#000000',
        tickfont_color='#000000'
    )
    
    return fig


def load_and_apply_styles():
    """Load and apply L'Oréal branded CSS styles"""
    styles = get_loreal_styles()
    st.markdown(styles, unsafe_allow_html=True)


# ================================
# DATA LOADING
# ================================
@st.cache_data
def load_dashboard_data():
    """Load the main dataset for the dashboard with caching"""
    data_paths = get_data_paths()
    df = load_data_with_fallback(
        data_paths['clustered_comments'], 
        "Clustered Comments Dataset"
    )
    
    if df is not None:
        # Basic data cleaning
        df = df.dropna(subset=['comment'] if 'comment' in df.columns else [])
        
        # Ensure consistent column naming
        if 'comment' not in df.columns and 'text' in df.columns:
            df['comment'] = df['text']
            
        st.success(f"✅ Dashboard data loaded: {len(df):,} records")
    
    return df


# ================================
# SIDEBAR CONTROLS
# ================================
def create_sidebar_filters(df):
    """Create sidebar filters for data exploration"""
    st.sidebar.title("🔍 Data Filters")
    
    # Initialize filtered dataframe
    filtered_df = df.copy()
    
    # Sentiment filter
    if "sentiment" in df.columns:
        sentiment_options = ["All"] + list(df["sentiment"].dropna().unique())
        selected_sentiment = st.sidebar.selectbox("Filter by Sentiment:", sentiment_options)
        if selected_sentiment != "All":
            filtered_df = filtered_df[filtered_df["sentiment"] == selected_sentiment]
    
    # Spam filter
    if "is_spam" in df.columns:
        spam_options = ["All", "Spam Only", "Non-Spam Only"]
        selected_spam = st.sidebar.selectbox("Filter by Spam Status:", spam_options)
        if selected_spam == "Spam Only":
            filtered_df = filtered_df[filtered_df["is_spam"].astype(str).str.lower().isin(["yes", "true", "spam"])]
        elif selected_spam == "Non-Spam Only":
            filtered_df = filtered_df[filtered_df["is_spam"].astype(str).str.lower().isin(["no", "false", "non-spam"])]
    
    # Quality score filter
    if "quality_score" in df.columns:
        min_quality, max_quality = float(df["quality_score"].min()), float(df["quality_score"].max())
        quality_range = st.sidebar.slider(
            "Quality Score Range:",
            min_value=min_quality,
            max_value=max_quality,
            value=(min_quality, max_quality),
            step=0.01
        )
        filtered_df = filtered_df[
            (filtered_df["quality_score"] >= quality_range[0]) & 
            (filtered_df["quality_score"] <= quality_range[1])
        ]
    
    # Sample size limiter for performance
    max_records = st.sidebar.slider(
        "Maximum Records to Display:",
        min_value=100,
        max_value=min(10000, len(filtered_df)),
        value=min(5000, len(filtered_df)),
        step=100
    )
    
    if len(filtered_df) > max_records:
        filtered_df = filtered_df.sample(n=max_records, random_state=42)
    
    # Display current filter info
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Current Selection:**")
    st.sidebar.markdown(f"📊 **Records**: {len(filtered_df):,}")
    if len(filtered_df) < len(df):
        st.sidebar.markdown(f"🔽 **Filtered from**: {len(df):,}")
    
    return filtered_df


# ================================
# MAIN APPLICATION
# ================================
def main():
    """Main application function"""
    
    # Apply L'Oréal branding and styles
    load_and_apply_styles()
    
    # Create header
    try:
        logo_base64 = encode_logo_to_base64("loreal-logo.jpeg")
    except:
        logo_base64 = None
        st.warning("⚠️ Logo file not found. Using text-only header.")
    
    header_html = create_header_html(logo_base64)
    st.markdown(header_html, unsafe_allow_html=True)
    
    # Load main dataset
    df = load_dashboard_data()
    
    if df is None or df.empty:
        st.error("❌ Could not load dataset. Please check your data files.")
        st.stop()
    
    # Create sidebar filters
    filtered_df = create_sidebar_filters(df)
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Overview", 
        "😊 Sentiment", 
        "📈 Trends", 
        "☁️ WordCloud", 
        "🤖 Classification", 
        "🏷️ Clusters", 
        "🛡️ Spam Detection"
    ])
    
    # Render individual tabs
    with tab1:
        render_overview_tab(filtered_df, apply_brand_style)
    
    with tab2:
        render_sentiment_tab(filtered_df, apply_brand_style)
    
    with tab3:
        render_trends_tab(filtered_df, apply_brand_style)
    
    with tab4:
        render_wordcloud_tab(filtered_df)
    
    with tab5:
        render_classification_tab()
    
    with tab6:
        render_cluster_tab(filtered_df, apply_brand_style)
    
    with tab7:
        render_spam_tab(filtered_df, apply_brand_style)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #6D6E70; font-size: 0.9rem; padding: 1rem 0;">
            <p><strong>TrendSpotter</strong> - L'Oréal AI-Powered Social Media Analytics Platform</p>
            <p>Built with ❤️ using Streamlit | Powered by BERT & PyTorch</p>
        </div>
        """, 
        unsafe_allow_html=True
    )


# ================================
# APPLICATION ENTRY POINT
# ================================
if __name__ == "__main__":
    main()