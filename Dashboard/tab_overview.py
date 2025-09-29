"""
Overview Tab Module for L'Oréal TrendSpotter Dashboard
Displays key metrics, charts, and summary statistics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
import os


def get_ai_insights(filtered_df):
    """
    Generate AI-powered business insights using OpenRouter API
    
    Args:
        filtered_df (pd.DataFrame): Filtered dataset
    
    Returns:
        dict: Dictionary containing insights for different sections
    """
    try:
        # Prepare data summary for AI analysis
        total_comments = len(filtered_df)
        avg_quality = filtered_df['quality_score'].mean() if 'quality_score' in filtered_df.columns else None
        avg_relevance = filtered_df['relevance_score'].mean() if 'relevance_score' in filtered_df.columns else None
        
        # Format quality and relevance for display
        quality_display = f"{avg_quality:.2f}" if avg_quality is not None else "N/A"
        relevance_display = f"{avg_relevance:.2f}" if avg_relevance is not None else "N/A"
        
        # Spam probability calculation
        spam_prob = 0
        if "is_spam" in filtered_df.columns:
            spam_data = filtered_df["is_spam"].astype(str).str.lower()
            spam_count = spam_data.isin(["yes", "true", "spam"]).sum()
            spam_prob = (spam_count / total_comments) if total_comments > 0 else 0
        
        # Top categories
        top_categories = {}
        if "new_cluster" in filtered_df.columns:
            top_categories = filtered_df["new_cluster"].value_counts().head(10).to_dict()
        
        # Quality and relevance distributions
        quality_dist = "N/A"
        relevance_dist = "N/A"
        if 'quality_score' in filtered_df.columns and avg_quality is not None:
            quality_dist = f"Mean: {avg_quality:.2f}, Min: {filtered_df['quality_score'].min():.2f}, Max: {filtered_df['quality_score'].max():.2f}, Std: {filtered_df['quality_score'].std():.2f}"
        if 'relevance_score' in filtered_df.columns and avg_relevance is not None:
            relevance_dist = f"Mean: {avg_relevance:.2f}, Min: {filtered_df['relevance_score'].min():.2f}, Max: {filtered_df['relevance_score'].max():.2f}, Std: {filtered_df['relevance_score'].std():.2f}"
        
        # Create prompt for AI analysis
        prompt = f"""
        As a business analyst for L'Oréal, analyze this social media/comments data and provide exactly 4 business insights (each within 100 words):

        DATA SUMMARY:
        - Total Comments: {total_comments:,}
        - Average Quality Score: {quality_display}
        - Average Relevance Score: {relevance_display}  
        - Spam Probability: {spam_prob:.1%}
        - Quality Distribution: {quality_dist}
        - Relevance Distribution: {relevance_dist}
        - Top 10 Categories: {dict(list(top_categories.items())[:5]) if top_categories else 'N/A'}...

        Please provide insights for:
        1. Key Metrics Overview (total comments, quality, relevance, spam)
        2. Quality Score Distribution Analysis
        3. Relevance Score Distribution Analysis  
        4. Top Categories Strategic Implications

        Format as: **Insight 1:** [content] **Insight 2:** [content] **Insight 3:** [content] **Insight 4:** [content]
        """
        
        # Get OpenRouter API key from environment or Streamlit secrets
        openrouter_key = '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'
        
        if not openrouter_key:
            return {
                "key_metrics": "🔑 AI insights require OpenRouter API key configuration.",
                "quality_distribution": "📊 Configure API key to unlock AI-powered quality analysis.",
                "relevance_distribution": "📈 Set OPENROUTER_API_KEY to get relevance insights.",
                "top_categories": "🎯 API key needed for strategic category analysis."
            }
        
        # Make API request to OpenRouter
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer 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",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://loreal-trendspotter.streamlit.app",
                "X-Title": "L'Oréal TrendSpotter Dashboard",
            },
            data=json.dumps({
                "model": "google/gemini-2.5-flash",
                "messages": [
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.7
            }),
            timeout=30
        )
        
        if response.status_code == 200:
            ai_response = response.json()
            content = ai_response['choices'][0]['message']['content']
            
            # Parse the response to extract individual insights
            insights = {}
            parts = content.split('**Insight ')
            
            if len(parts) >= 5:  # Should have 4 insights
                insights["key_metrics"] = parts[1].split(':**')[1].split('**Insight')[0].strip()
                insights["quality_distribution"] = parts[2].split(':**')[1].split('**Insight')[0].strip()
                insights["relevance_distribution"] = parts[3].split(':**')[1].split('**Insight')[0].strip()
                insights["top_categories"] = parts[4].split(':**')[1].strip()
            else:
                # Fallback if parsing fails
                insights = {
                    "key_metrics": content[:200] + "...",
                    "quality_distribution": "Quality analysis insights generated by AI.",
                    "relevance_distribution": "Relevance analysis insights generated by AI.", 
                    "top_categories": "Category strategic implications from AI analysis."
                }
                
            return insights
        else:
            return {
                "key_metrics": f"⚠️ AI service temporarily unavailable (Status: {response.status_code})",
                "quality_distribution": "📊 Unable to generate quality insights at this time.",
                "relevance_distribution": "📈 Relevance analysis currently unavailable.",
                "top_categories": "🎯 Category insights service unavailable."
            }
            
    except Exception as e:
        return {
            "key_metrics": f"🔧 AI insights error: {str(e)[:50]}...",
            "quality_distribution": "📊 Technical issue generating quality insights.",
            "relevance_distribution": "📈 Error in relevance analysis generation.",
            "top_categories": "🎯 Unable to generate category insights."
        }


def render_overview_tab(filtered_df, apply_brand_style):
    """
    Render the Overview tab with key metrics and visualizations
    
    Args:
        filtered_df (pd.DataFrame): Filtered dataset
        apply_brand_style (function): Function to apply L'Oréal brand styling to plots
    """
    # Generate AI insights
    with st.spinner("🤖 Generating AI-powered business insights..."):
        ai_insights = get_ai_insights(filtered_df)
    
    # ---- TITLE ----
    st.markdown('<h3 style="color:#000;">Key Metrics</h3>', unsafe_allow_html=True)

    # ---- FOUR CARDS ----
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #FFFFFF 0%, #F8F9FA 100%);
                border: 2px solid #E5E7EB;
                border-radius: 16px;
                padding: 24px 20px;
                margin: 12px 0;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
                text-align: center;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            ">
                <div style="
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    height: 4px;
                    background: linear-gradient(90deg, #ED1B2E 0%, #FF6B7A 100%);
                "></div>
                <div style="
                    margin-top: 8px;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                ">
                    <div style="
                        font-size: 14px;
                        font-weight: 600;
                        color: #6B7280;
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                        margin-bottom: 8px;
                    ">Total Comments</div>
                    <div style="
                        font-size: 36px;
                        font-weight: 700;
                        color: #111827;
                        line-height: 1;
                        margin-bottom: 4px;
                    ">{len(filtered_df):,}</div>
                    <div style="
                        font-size: 12px;
                        color: #9CA3AF;
                        font-weight: 500;
                    ">Active Records</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        _render_quality_metric_card(filtered_df)

    with col3:
        _render_relevance_metric_card(filtered_df)

    with col4:
        _render_spam_probability_card(filtered_df)

    # AI Insights for Key Metrics
    st.markdown("### 🤖 AI Business Insights - Key Metrics")
    st.info(ai_insights["key_metrics"])

    st.divider()

    # Quality Score and Relevance Score Histograms side-by-side with improved sizing
    hist_col1, hist_col2 = st.columns(2)
    
    with hist_col1:
        if "quality_score" in filtered_df.columns:
            st.markdown('<h4 style="color:#000;">Quality Score Distribution</h4>', unsafe_allow_html=True)
            fig_quality = px.histogram(
                filtered_df, 
                x="quality_score", 
                nbins=25,  # Reduced bins for better readability in smaller space
                title="Quality Score Distribution",
                height=450  # Increased height to compensate for narrower width
            )
            apply_brand_style(fig_quality)
            # Update layout for better spacing in column
            fig_quality.update_layout(
                bargap=0.15,
                xaxis_title="Quality Score",
                yaxis_title="Count",
                showlegend=False,
                margin=dict(l=40, r=40, t=60, b=40),  # Adjust margins
                font=dict(size=10)  # Smaller font for better fit
            )
            st.plotly_chart(fig_quality, use_container_width=True, key="overview-quality-hist")
    
    with hist_col2:
        if "relevance_score" in filtered_df.columns:
            st.markdown('<h4 style="color:#000;">Relevance Score Distribution</h4>', unsafe_allow_html=True)
            fig_relevance = px.histogram(
                filtered_df, 
                x="relevance_score", 
                nbins=25,  # Reduced bins for better readability in smaller space
                title="Relevance Score Distribution",
                height=450  # Increased height to compensate for narrower width
            )
            apply_brand_style(fig_relevance)
            # Update layout for better spacing in column
            fig_relevance.update_layout(
                bargap=0.15,
                xaxis_title="Relevance Score",
                yaxis_title="Count",
                showlegend=False,
                margin=dict(l=40, r=40, t=60, b=40),  # Adjust margins
                font=dict(size=10)  # Smaller font for better fit
            )
            st.plotly_chart(fig_relevance, use_container_width=True, key="overview-relevance-hist")

    # AI Insights for Distribution Analysis
    dist_col1, dist_col2 = st.columns(2)
    with dist_col1:
        st.markdown("### 🤖 Quality Distribution Insights")
        st.info(ai_insights["quality_distribution"])
    
    with dist_col2:
        st.markdown("### 🤖 Relevance Distribution Insights") 
        st.info(ai_insights["relevance_distribution"])
    
    st.divider()
    
    # Top 10 Clusters/Categories Bar Chart
    _render_top_categories(filtered_df, apply_brand_style)
    
    # AI Insights for Top Categories
    st.markdown("### 🤖 Strategic Category Insights")
    st.info(ai_insights["top_categories"])


def _render_quality_metric_card(filtered_df):
    """Render the quality metric card"""
    if "quality_score" in filtered_df.columns:
        quality_value = filtered_df['quality_score'].mean()
        quality_color = "#10B981" if quality_value >= 0.7 else "#F59E0B" if quality_value >= 0.5 else "#EF4444"
        quality_label = "Excellent" if quality_value >= 0.7 else "Good" if quality_value >= 0.5 else "Needs Improvement"
        
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #FFFFFF 0%, #F8F9FA 100%);
                border: 2px solid #E5E7EB;
                border-radius: 16px;
                padding: 24px 20px;
                margin: 12px 0;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
                text-align: center;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            ">
                <div style="
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    height: 4px;
                    background: linear-gradient(90deg, {quality_color} 0%, {quality_color}80 100%);
                "></div>
                <div style="
                    margin-top: 8px;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                ">
                    <div style="
                        font-size: 14px;
                        font-weight: 600;
                        color: #6B7280;
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                        margin-bottom: 8px;
                    ">Average Quality</div>
                    <div style="
                        font-size: 36px;
                        font-weight: 700;
                        color: #111827;
                        line-height: 1;
                        margin-bottom: 4px;
                    ">{quality_value:.2f}</div>
                    <div style="
                        font-size: 12px;
                        color: {quality_color};
                        font-weight: 600;
                        background: {quality_color}15;
                        padding: 4px 12px;
                        border-radius: 12px;
                    ">{quality_label}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        _render_no_data_card("Average Quality")


def _render_relevance_metric_card(filtered_df):
    """Render the relevance metric card"""
    if "relevance_score" in filtered_df.columns:
        relevance_value = filtered_df['relevance_score'].mean()
        relevance_color = "#10B981" if relevance_value >= 0.7 else "#F59E0B" if relevance_value >= 0.5 else "#EF4444"
        relevance_label = "High" if relevance_value >= 0.7 else "Medium" if relevance_value >= 0.5 else "Low"
        
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #FFFFFF 0%, #F8F9FA 100%);
                border: 2px solid #E5E7EB;
                border-radius: 16px;
                padding: 24px 20px;
                margin: 12px 0;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
                text-align: center;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            ">
                <div style="
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    height: 4px;
                    background: linear-gradient(90deg, {relevance_color} 0%, {relevance_color}80 100%);
                "></div>
                <div style="
                    margin-top: 8px;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                ">
                    <div style="
                        font-size: 14px;
                        font-weight: 600;
                        color: #6B7280;
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                        margin-bottom: 8px;
                    ">Average Relevance</div>
                    <div style="
                        font-size: 36px;
                        font-weight: 700;
                        color: #111827;
                        line-height: 1;
                        margin-bottom: 4px;
                    ">{relevance_value:.2f}</div>
                    <div style="
                        font-size: 12px;
                        color: {relevance_color};
                        font-weight: 600;
                        background: {relevance_color}15;
                        padding: 4px 12px;
                        border-radius: 12px;
                    ">{relevance_label} Relevance</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        _render_no_data_card("Average Relevance")


def _render_spam_probability_card(filtered_df):
    """Render the spam probability metric card"""
    if "is_spam" in filtered_df.columns:
        # Calculate spam probability
        spam_data = filtered_df["is_spam"].astype(str).str.lower()
        spam_count = spam_data.isin(["yes", "true", "spam"]).sum()
        total_count = len(filtered_df)
        spam_probability = (spam_count / total_count) if total_count > 0 else 0
        
        # Determine color and label based on spam probability
        if spam_probability <= 0.1:
            spam_color = "#10B981"  # Green for low spam
            spam_label = "Low Risk"
        elif spam_probability <= 0.3:
            spam_color = "#F59E0B"  # Yellow for medium spam
            spam_label = "Medium Risk"
        else:
            spam_color = "#EF4444"  # Red for high spam
            spam_label = "High Risk"
        
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #FFFFFF 0%, #F8F9FA 100%);
                border: 2px solid #E5E7EB;
                border-radius: 16px;
                padding: 24px 20px;
                margin: 12px 0;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
                text-align: center;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            ">
                <div style="
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    height: 4px;
                    background: linear-gradient(90deg, {spam_color} 0%, {spam_color}80 100%);
                "></div>
                <div style="
                    margin-top: 8px;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                ">
                    <div style="
                        font-size: 14px;
                        font-weight: 600;
                        color: #6B7280;
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                        margin-bottom: 8px;
                    ">Spam Probability</div>
                    <div style="
                        font-size: 36px;
                        font-weight: 700;
                        color: #111827;
                        line-height: 1;
                        margin-bottom: 4px;
                    ">{spam_probability:.1%}</div>
                    <div style="
                        font-size: 12px;
                        color: {spam_color};
                        font-weight: 600;
                        background: {spam_color}15;
                        padding: 4px 12px;
                        border-radius: 12px;
                    ">{spam_label}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        _render_no_data_card("Spam Probability")


def _render_no_data_card(title):
    """Render a no data available card"""
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #FFFFFF 0%, #F8F9FA 100%);
            border: 2px solid #E5E7EB;
            border-radius: 16px;
            padding: 24px 20px;
            margin: 12px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            text-align: center;
            position: relative;
            overflow: hidden;
        ">
            <div style="
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: linear-gradient(90deg, #9CA3AF 0%, #D1D5DB 100%);
            "></div>
            <div style="
                margin-top: 8px;
                display: flex;
                flex-direction: column;
                align-items: center;
            ">
                <div style="
                    font-size: 14px;
                    font-weight: 600;
                    color: #6B7280;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                    margin-bottom: 8px;
                ">{title}</div>
                <div style="
                    font-size: 36px;
                    font-weight: 700;
                    color: #9CA3AF;
                    line-height: 1;
                    margin-bottom: 4px;
                ">N/A</div>
                <div style="
                    font-size: 12px;
                    color: #9CA3AF;
                    font-weight: 500;
                ">No Data Available</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_top_categories(filtered_df, apply_brand_style):
    """Render top 10 categories for various cluster columns"""
    for col in ["new_cluster", "cluster", "predicted_category"]:
        if col in filtered_df.columns:
            st.markdown(f'<h3 style="color:#000;">Top 10 {col} Categories</h3>', unsafe_allow_html=True)
            cat_counts = filtered_df[col].value_counts().head(10).reset_index()
            cat_counts.columns = [col, "count"]
            
            # Create horizontal bar chart for better readability
            fig_cat = px.bar(
                cat_counts, 
                x="count", 
                y=col, 
                orientation='h',  # Horizontal bars
                title=f"Top 10 {col} Categories",
                height=500  # Fixed height for better aspect ratio
            )
            apply_brand_style(fig_cat)
            
            # Improve layout for horizontal bar chart
            fig_cat.update_layout(
                xaxis_title="Count",
                yaxis_title=col.replace('_', ' ').title(),
                showlegend=False,
                margin=dict(l=150, r=40, t=60, b=40),  # More left margin for labels
                yaxis=dict(
                    autorange="reversed",  # Show highest values at top
                    tickmode='linear'
                ),
                bargap=0.3  # Add space between bars
            )
            
            st.plotly_chart(fig_cat, use_container_width=True, key=f"overview-top10-{col}")