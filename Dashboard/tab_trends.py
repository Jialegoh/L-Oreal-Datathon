"""
Trends Analysis Tab Module for L'Oréal TrendSpotter Dashboard
Handles time-based analysis and trend visualization
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import os


def render_trends_tab(filtered_df, apply_brand_style):
    """Render the Trends Analysis tab with comprehensive time-based analytics"""
    st.markdown('<h3 style="color:#000;">📈 Trends Analysis</h3>', unsafe_allow_html=True)
    
    # Skip date column selection - use pre-processed time series data instead
    date_col = None  # Not needed since we use pre-processed CSV data
    
    try:
        # Use empty dataframe since we rely on pre-processed CSV data
        df_trends = pd.DataFrame()
        
        # Create tabs for different trend analyses
        trend_tab1, trend_tab2, trend_tab3 = st.tabs([
            "📊 Volume Trends", 
            "💭 Sentiment Trends", 
            "📈 Advanced Analytics"
        ])
        
        with trend_tab1:
            render_volume_trends(None, apply_brand_style)
        
        with trend_tab2:
            render_sentiment_trends(None, apply_brand_style)
        
        with trend_tab3:
            render_advanced_analytics(None, apply_brand_style)
            
    except Exception as e:
        st.error(f"❌ Error creating trends analysis: {str(e)}")
        st.info("Please check your data format and try again.")


def _resolve_ts_path(filename: str) -> Path:
    """Resolve TimeSeries CSV path with sensible fallbacks.

    Order tried:
    1) Env var LOREAL_TS_DIR/filename (if set)
    2) ../AI_Model/TimeSeries/filename relative to this file
    3) C:/L-Oreal-Datathon/AI_Model/TimeSeries/filename (Windows default)
    """
    # 1) Environment variable override
    ts_dir = os.getenv("LOREAL_TS_DIR")
    if ts_dir:
        candidate = Path(ts_dir) / filename
        if candidate.exists():
            return candidate

    # 2) Relative to repo (Dashboard/..)
    dashboard_dir = Path(__file__).parent
    candidate2 = (dashboard_dir / ".." / "AI_Model" / "TimeSeries" / filename).resolve()
    if candidate2.exists():
        return candidate2

    # 3) Default C: location as a last resort
    candidate3 = Path("C:/L-Oreal-Datathon/AI_Model/TimeSeries") / filename
    return candidate3

def prepare_trends_data(df, date_col):
    """Prepare and clean data for trend analysis"""
    
    with st.spinner("🔄 Preparing trend data..."):
        df_trends = df.copy()
        
        # Convert to datetime
        df_trends[date_col] = pd.to_datetime(df_trends[date_col], errors='coerce')
        
        # Remove invalid dates
        original_count = len(df_trends)
        df_trends = df_trends.dropna(subset=[date_col])
        cleaned_count = len(df_trends)
        
        if cleaned_count < original_count:
            st.info(f"ℹ️ Removed {original_count - cleaned_count:,} rows with invalid dates")
        
        # Extract time components
        df_trends['date'] = df_trends[date_col].dt.date
        df_trends['year'] = df_trends[date_col].dt.year
        df_trends['month'] = df_trends[date_col].dt.month
        df_trends['day'] = df_trends[date_col].dt.day
        df_trends['weekday'] = df_trends[date_col].dt.day_name()
        df_trends['hour'] = df_trends[date_col].dt.hour
        df_trends['week'] = df_trends[date_col].dt.isocalendar().week
        
        # Sort by date
        df_trends = df_trends.sort_values(date_col)
    
    return df_trends


def display_date_range_info(df_trends):
    """Display date range information"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📅 Total Days", len(df_trends['date'].unique()))
    
    with col2:
        min_date = df_trends['date'].min()
        st.metric("🗓️ Start Date", min_date.strftime('%Y-%m-%d'))
    
    with col3:
        max_date = df_trends['date'].max()
        st.metric("🗓️ End Date", max_date.strftime('%Y-%m-%d'))
    
    with col4:
        date_range = (max_date - min_date).days
        st.metric("📊 Date Range", f"{date_range} days")
    
    st.divider()


@st.cache_data
def load_time_series_data():
    """Load pre-processed time series data from CSV"""
    try:
        data_path = _resolve_ts_path("time_series_cluster_analysis.csv")
        if data_path.exists():
            df = pd.read_csv(data_path)
            
            # Validate required columns (including sentiment columns)
            required_cols = ['new_cluster', 'year', 'month', 'count']
            sentiment_cols = ['sentiment_positive', 'sentiment_negative', 'sentiment_neutral']
            all_required_cols = required_cols + sentiment_cols
            missing_cols = [col for col in all_required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns in time series data: {missing_cols}")
                st.info(f"Available columns: {list(df.columns)}")
                return None
            

            
            return df
        else:
            st.warning(f"⚠️ Time series file not found: {data_path}")
            return None
    except Exception as e:
        st.error(f"❌ Error loading time series data: {str(e)}")
        return None


def render_volume_trends(df_trends, apply_brand_style):
    """Render volume trend analysis using pre-processed time series data"""
    
    st.markdown("### 📊 Comment Volume Trends")
    
    # Load time series data
    ts_data = load_time_series_data()
    
    if ts_data is None:
        st.warning("⚠️ Time series data not available. Using filtered data instead.")
        render_volume_trends_fallback(df_trends, apply_brand_style)
        return
    
    # Display key volume insights first
    st.markdown("#### 📊 Key Volume Insights")
    
    # Calculate overall statistics
    overall_data = ts_data.groupby(['year', 'month'])['count'].sum().reset_index()
    overall_data['period'] = pd.to_datetime(overall_data[['year', 'month']].assign(day=1))
    overall_data = overall_data.sort_values('period')
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_volume = overall_data['count'].mean()
        st.metric("📊 Average Volume", f"{avg_volume:,.0f}")
    
    with col2:
        peak_volume = overall_data['count'].max()
        peak_idx = overall_data['count'].idxmax()
        peak_date = overall_data.loc[peak_idx, 'period']
        st.metric("🔝 Peak Volume", f"{peak_volume:,}")
        st.caption(f"in {peak_date.strftime('%Y-%m')}")
    
    with col3:
        total_volume = overall_data['count'].sum()
        st.metric("📈 Total Comments", f"{total_volume:,}")
    
    with col4:
        # Calculate growth rate
        if len(overall_data) > 1:
            first_period = overall_data.iloc[0]['count']
            last_period = overall_data.iloc[-1]['count']
            growth_rate = ((last_period - first_period) / first_period) * 100 if first_period > 0 else 0
            growth_direction = "📈" if growth_rate > 0 else "📉" if growth_rate < 0 else "➡️"
            st.metric("📊 Growth Rate", f"{growth_direction} {abs(growth_rate):.1f}%")
        else:
            st.metric("📊 Growth Rate", "N/A")
    
    st.divider()
    
    # Time period selection (only Monthly and Yearly available)
    time_period = st.selectbox("Select time period:", 
                              ["Monthly", "Yearly"], 
                              key="volume_period")
    
    # Process time series data based on selected period
    if time_period == "Monthly":
        # Group by year and month, sum all clusters
        trend_data = ts_data.groupby(['year', 'month'])['count'].sum().reset_index()
        trend_data['period'] = pd.to_datetime(trend_data[['year', 'month']].assign(day=1))
        title = "Monthly Comment Volume (All Categories)"
        period_format = '%Y-%m'
        
    else:  # Yearly
        # Group by year only, sum all clusters and months
        trend_data = ts_data.groupby('year')['count'].sum().reset_index()
        trend_data['period'] = pd.to_datetime(trend_data['year'], format='%Y')
        title = "Yearly Comment Volume (All Categories)"
        period_format = '%Y'
    
    # Sort by period
    trend_data = trend_data.sort_values('period')
    
    if trend_data.empty:
        st.warning("⚠️ No valid trend data to display.")
        return
    
    # Create line chart with proper height constraints
    fig_volume = px.line(trend_data, x='period', y='count', 
                        title=title,
                        labels={'period': 'Time Period', 'count': 'Number of Comments'},
                        height=450)  # Fixed height to prevent stretching
    
    # Add markers for better visibility
    fig_volume.update_traces(mode='lines+markers', marker=dict(size=8))
    
    # Customize the chart with proper margins
    fig_volume.update_layout(
        xaxis_title='Time Period',
        yaxis_title='Number of Comments',
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),  # Proper margins
        font=dict(size=12)  # Consistent font size
    )
    
    apply_brand_style(fig_volume)
    st.plotly_chart(fig_volume, use_container_width=True, key="volume-trends-chart")
    
    # Show top categories breakdown
    render_category_breakdown(ts_data, time_period, apply_brand_style)


def render_category_breakdown(ts_data, time_period, apply_brand_style):
    """Render category breakdown for the selected time period"""
    
    st.markdown("### 🏷️ Top Categories Breakdown")
    
    try:
        # Debug: Show data info
        st.info(f"📊 Processing {len(ts_data)} time series records...")
        
        # Get top categories overall with better error handling
        if ts_data.empty or 'new_cluster' not in ts_data.columns or 'count' not in ts_data.columns:
            st.warning("⚠️ Required columns 'new_cluster' or 'count' not found in time series data.")
            st.info(f"Available columns: {list(ts_data.columns)}")
            return
            
        top_categories = ts_data.groupby('new_cluster')['count'].sum().sort_values(ascending=False).head(10)
        
        if top_categories.empty:
            st.warning("⚠️ No category data available for breakdown.")
            return
            
        st.success(f"✅ Found {len(top_categories)} categories for analysis")
        
        # Filter data for top categories
        top_category_data = ts_data[ts_data['new_cluster'].isin(top_categories.index)]
        
        if time_period == "Monthly":
            # Group by year, month, and category
            category_trends = top_category_data.groupby(['year', 'month', 'new_cluster'])['count'].sum().reset_index()
            category_trends['period'] = pd.to_datetime(category_trends[['year', 'month']].assign(day=1))
            title = "Monthly Trends by Top 10 Categories"
            
        else:  # Yearly
            # Group by year and category
            category_trends = top_category_data.groupby(['year', 'new_cluster'])['count'].sum().reset_index()
            category_trends['period'] = pd.to_datetime(category_trends['year'], format='%Y')
            title = "Yearly Trends by Top 10 Categories"
    
        # Create line chart for categories with proper height constraints
        fig_categories = px.line(category_trends, x='period', y='count', color='new_cluster',
                               title=title,
                               labels={'period': 'Time Period', 'count': 'Number of Comments', 'new_cluster': 'Category'},
                               height=500)  # Fixed height to prevent stretching
        
        # Add markers and improve layout
        fig_categories.update_traces(mode='lines+markers', marker=dict(size=4))
        fig_categories.update_layout(
            margin=dict(l=40, r=40, t=60, b=40),
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.05
            )
        )
        
        apply_brand_style(fig_categories)
        st.plotly_chart(fig_categories, use_container_width=True, key="category-trends-chart")
        
        # Show top categories table with improved error handling
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Top 10 Categories (Total Volume)")
            try:
                # Create dataframe with better error handling
                top_categories_df = top_categories.reset_index()
                top_categories_df.columns = ['Category', 'Total Comments']
                
                # Ensure we have data
                if len(top_categories_df) == 0:
                    st.warning("⚠️ No category data to display")
                else:
                    # Calculate percentages
                    total_sum = top_categories_df['Total Comments'].sum()
                    if total_sum > 0:
                        top_categories_df['Percentage'] = (top_categories_df['Total Comments'] / total_sum * 100).round(1)
                        top_categories_df['Percentage'] = top_categories_df['Percentage'].astype(str) + '%'
                    else:
                        top_categories_df['Percentage'] = '0%'
                    
                    # Display with better formatting
                    top_categories_df['Total Comments'] = top_categories_df['Total Comments'].apply(lambda x: f"{x:,}")
                    
                    # Create clean table with index starting from 1
                    display_df = top_categories_df.copy()
                    
                    # Set index to start from 1 instead of 0
                    display_df.index = range(1, len(display_df) + 1)
                    
                    # Display the clean table using st.table (will use global minimalist CSS styling)
                    st.table(display_df)
                    
            except Exception as table_error:
                st.error(f"❌ Error creating categories table: {str(table_error)}")
                # Fallback display
                st.write("**Raw data:**")
                st.write(top_categories.head(10))
        
        with col2:
            # Category distribution pie chart with proper height
            try:
                # Recreate dataframe for pie chart
                pie_data = top_categories.reset_index()
                pie_data.columns = ['Category', 'Total Comments']
                
                fig_pie = px.pie(pie_data, values='Total Comments', names='Category',
                                title='Category Distribution (Top 10)',
                                height=400)  # Fixed height to prevent stretching
                
                fig_pie.update_layout(
                    margin=dict(l=20, r=20, t=60, b=20),
                    showlegend=True,
                    legend=dict(
                        orientation="v",
                        yanchor="middle",
                        y=0.5,
                        xanchor="left",
                        x=1.05
                    )
                )
                
                apply_brand_style(fig_pie)
                st.plotly_chart(fig_pie, use_container_width=True, key="category-pie-chart")
                
            except Exception as pie_error:
                st.error(f"❌ Error creating pie chart: {str(pie_error)}")
                
    except Exception as e:
        st.error(f"❌ Error in category breakdown: {str(e)}")
        st.info("Please check your time series data format and try again.")


def render_volume_trends_fallback(df_trends, apply_brand_style):
    """Fallback method using filtered dataframe when time series data is not available"""
    
    st.info("ℹ️ Using filtered dataset for trend analysis.")
    
    # Time period selection
    time_period = st.selectbox("Select time period:", 
                              ["Daily", "Weekly", "Monthly", "Yearly"], 
                              key="volume_period_fallback")
    
    # Group data based on selected period
    if time_period == "Daily":
        trend_data = df_trends.groupby('date').size().reset_index(name='count')
        trend_data['period'] = trend_data['date']
        title = "Daily Comment Volume"
    elif time_period == "Weekly":
        trend_data = df_trends.groupby(['year', 'week']).size().reset_index(name='count')
        trend_data['period'] = pd.to_datetime(trend_data[['year', 'week']].assign(day=1), format='%Y %W %w', errors='coerce')
        title = "Weekly Comment Volume"
    elif time_period == "Monthly":
        trend_data = df_trends.groupby(['year', 'month']).size().reset_index(name='count')
        trend_data['period'] = pd.to_datetime(trend_data[['year', 'month']].assign(day=1))
        title = "Monthly Comment Volume"
    else:  # Yearly
        trend_data = df_trends.groupby('year').size().reset_index(name='count')
        trend_data['period'] = pd.to_datetime(trend_data['year'], format='%Y')
        title = "Yearly Comment Volume"
    
    # Remove any NaT values that might have been created
    trend_data = trend_data.dropna(subset=['period'])
    
    if trend_data.empty:
        st.warning("⚠️ No valid trend data to display.")
        return
    
    # Create line chart
    fig_volume = px.line(trend_data, x='period', y='count', 
                        title=title,
                        labels={'period': 'Time Period', 'count': 'Number of Comments'})
    
    # Add markers for better visibility
    fig_volume.update_traces(mode='lines+markers', marker=dict(size=6))
    
    apply_brand_style(fig_volume)
    st.plotly_chart(fig_volume, use_container_width=True)
    
    # Volume statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_volume = trend_data['count'].mean()
        st.metric("📊 Average Volume", f"{avg_volume:.1f}")
    
    with col2:
        peak_volume = trend_data['count'].max()
        peak_idx = trend_data['count'].idxmax()
        peak_date = trend_data.loc[peak_idx, 'period']
        st.metric("🔝 Peak Volume", f"{peak_volume}")
        if hasattr(peak_date, 'strftime'):
            st.caption(f"on {peak_date.strftime('%Y-%m-%d')}")
    
    with col3:
        total_volume = trend_data['count'].sum()
        st.metric("📈 Total Comments", f"{total_volume:,}")


def render_sentiment_trends(df_trends, apply_brand_style):
    """Render sentiment trend analysis using time series data with sentiment columns"""
    
    st.markdown("### 💭 Sentiment Trends Over Time")
    
    # Load time series data with sentiment columns
    ts_data = load_time_series_data()
    
    if ts_data is None:
        st.warning("⚠️ Time series data not available for sentiment analysis.")
        return
    
    # Check for required sentiment columns
    required_sentiment_cols = ['sentiment_positive', 'sentiment_negative', 'sentiment_neutral']
    missing_cols = [col for col in required_sentiment_cols if col not in ts_data.columns]
    
    if missing_cols:
        st.warning(f"⚠️ Missing sentiment columns: {missing_cols}")
        st.info(f"Available columns: {list(ts_data.columns)}")
        return
    
    # Display sentiment statistics at the top for better visibility
    st.markdown("#### 💭 Key Sentiment Insights")
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate totals across all periods for display
    total_positive = ts_data['sentiment_positive'].sum()
    total_negative = ts_data['sentiment_negative'].sum()
    total_neutral = ts_data['sentiment_neutral'].sum()
    total_comments = total_positive + total_negative + total_neutral
    
    with col1:
        st.metric("😊 Total Positive", f"{total_positive:,}")
    
    with col2:
        st.metric("😞 Total Negative", f"{total_negative:,}")
    
    with col3:
        st.metric("😐 Total Neutral", f"{total_neutral:,}")
    
    with col4:
        if total_comments > 0:
            positive_pct = (total_positive / total_comments) * 100
            st.metric("📊 Positive %", f"{positive_pct:.1f}%")
        else:
            st.metric("📊 Positive %", "N/A")
    
    st.divider()
    
    # Time grouping for sentiment trends (only Monthly and Yearly)
    time_grouping = st.selectbox("Time grouping:", 
                                ["Monthly", "Yearly"], 
                                key="sentiment_grouping")
    
    # Process time series data based on selected period
    if time_grouping == "Monthly":
        # Group by year and month, sum sentiment columns across all clusters
        sentiment_summary = ts_data.groupby(['year', 'month'])[required_sentiment_cols].sum().reset_index()
        sentiment_summary['period'] = pd.to_datetime(sentiment_summary[['year', 'month']].assign(day=1))
        title_suffix = "Monthly"
        period_format = '%Y-%m'
        
    else:  # Yearly
        # Group by year only, sum sentiment columns across all clusters and months
        sentiment_summary = ts_data.groupby('year')[required_sentiment_cols].sum().reset_index()
        sentiment_summary['period'] = pd.to_datetime(sentiment_summary['year'], format='%Y')
        title_suffix = "Yearly"
        period_format = '%Y'
    
    # Sort by period
    sentiment_summary = sentiment_summary.sort_values('period')
    
    # Convert to long format for plotting
    sentiment_trends = pd.melt(
        sentiment_summary, 
        id_vars=['period'], 
        value_vars=required_sentiment_cols,
        var_name='sentiment_type', 
        value_name='count'
    )
    
    # Clean up sentiment type names for display
    sentiment_trends['sentiment'] = sentiment_trends['sentiment_type'].str.replace('sentiment_', '').str.title()
    
    if sentiment_trends.empty:
        st.warning("⚠️ No valid sentiment trend data to display.")
        return
    
    # Sentiment trend line chart with proper colors
    fig_sentiment = px.line(sentiment_trends, x='period', y='count', color='sentiment',
                           title=f'{title_suffix} Sentiment Trends (All Categories)',
                           labels={'period': 'Time Period', 'count': 'Number of Comments'},
                           height=450,
                           color_discrete_map={
                               'Positive': '#10B981',
                               'Negative': '#EF4444', 
                               'Neutral': '#F59E0B'
                           })
    
    # Add markers
    fig_sentiment.update_traces(mode='lines+markers', marker=dict(size=6))
    
    # Update layout
    fig_sentiment.update_layout(
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=True
    )
    
    apply_brand_style(fig_sentiment)
    st.plotly_chart(fig_sentiment, use_container_width=True, key="sentiment-trends-chart")
    
    # Sentiment distribution over time (stacked area)
    try:
        # Create pivot table for stacked area chart
        sentiment_pivot = sentiment_trends.pivot(index='period', columns='sentiment', values='count').fillna(0)
        
        if not sentiment_pivot.empty:
            fig_area = go.Figure()
            
            colors = {'Positive': '#10B981', 'Negative': '#EF4444', 'Neutral': '#F59E0B'}
            
            # Add traces in specific order for better stacking
            for sentiment in ['Positive', 'Neutral', 'Negative']:
                if sentiment in sentiment_pivot.columns:
                    fig_area.add_trace(go.Scatter(
                        x=sentiment_pivot.index, 
                        y=sentiment_pivot[sentiment],
                        mode='lines',
                        stackgroup='one',
                        name=sentiment,
                        fill='tonexty',
                        line=dict(color=colors[sentiment], width=0),
                        fillcolor=colors[sentiment]
                    ))
            
            fig_area.update_layout(
                title=f'{title_suffix} Sentiment Distribution (Stacked Area)',
                xaxis_title='Time Period',
                yaxis_title='Number of Comments',
                hovermode='x unified',
                height=450,
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            apply_brand_style(fig_area)
            st.plotly_chart(fig_area, use_container_width=True, key="sentiment-area-chart")
            
    except Exception as e:
        st.warning(f"⚠️ Could not create stacked area chart: {str(e)}")
    



def render_advanced_analytics(df_trends, apply_brand_style):
    """Render advanced trend analytics"""
    
    st.markdown("### 📈 Advanced Trend Analytics")
    
    # Load analytics data first for statistics
    hourly_data = None
    weekday_data = None
    daily_counts = None
    
    try:
        # Load hourly patterns from CSV
        hourly_csv_path = _resolve_ts_path("hourly_comment_patterns.csv")
        hourly_data = pd.read_csv(hourly_csv_path)
    except:
        pass
    
    try:
        # Load weekday patterns from CSV
        weekday_csv_path = _resolve_ts_path("weekday_comment_patterns.csv")
        weekday_data = pd.read_csv(weekday_csv_path)
    except:
        pass
    
    try:
        # Load daily patterns from CSV
        daily_csv_path = _resolve_ts_path("daily_comment_patterns.csv")
        daily_data = pd.read_csv(daily_csv_path)
        daily_data['date'] = pd.to_datetime(daily_data['date'])
        daily_counts = daily_data.sort_values('date')
    except:
        st.warning("⚠️ Daily patterns CSV not found. Advanced analytics unavailable.")
        return
    
    # Display key statistics at the top
    st.markdown("#### 📊 Key Activity Insights")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        try:
            if hourly_data is not None and not hourly_data.empty:
                peak_hour_idx = hourly_data['comment_count'].idxmax()
                peak_hour_label = hourly_data.loc[peak_hour_idx, 'hour_label']
                peak_hour_count = hourly_data.loc[peak_hour_idx, 'comment_count']
                st.metric("🕐 Peak Hour", peak_hour_label, f"{peak_hour_count:,} comments")
            else:
                st.metric("🕐 Peak Hour", "N/A", "Data unavailable")
        except:
            st.metric("🕐 Peak Hour", "N/A", "Data unavailable")
    
    with col2:
        try:
            if weekday_data is not None and not weekday_data.empty:
                peak_day_idx = weekday_data['comment_count'].idxmax()
                peak_day = weekday_data.loc[peak_day_idx, 'weekday']
                peak_day_count = weekday_data.loc[peak_day_idx, 'comment_count']
                st.metric("📅 Peak Day", peak_day, f"{peak_day_count:,} comments")
            else:
                st.metric("📅 Peak Day", "N/A", "Data unavailable")
        except:
            st.metric("📅 Peak Day", "N/A", "Data unavailable")
    
    with col3:
        try:
            avg_daily = daily_counts['comment_count'].mean()
            std_daily = daily_counts['comment_count'].std()
            st.metric("📊 Daily Avg", f"{avg_daily:.1f}", f"±{std_daily:.1f}")
        except:
            st.metric("📊 Daily Avg", "N/A", "Data unavailable")
    
    with col4:
        try:
            # Calculate trend direction (simple linear regression slope)
            if len(daily_counts) > 1:
                x = np.arange(len(daily_counts))
                slope = np.polyfit(x, daily_counts['comment_count'], 1)[0]
                trend_direction = "📈 Rising" if slope > 0 else "📉 Falling" if slope < 0 else "➡️ Stable"
                st.metric("📈 Trend", trend_direction, f"{slope:.2f}/day")
            else:
                st.metric("📈 Trend", "N/A", "Insufficient data")
        except:
            st.metric("📈 Trend", "N/A", "Data unavailable")
    
    st.divider()
    
    # Hourly patterns
    st.markdown("#### ⏰ Hourly Activity Patterns")
    
    try:
        if hourly_data is not None and not hourly_data.empty:
            # Create line chart using hour_label and comment_count
            fig_hourly = px.line(hourly_data, x='hour_label', y='comment_count',
                               title='Comments by Hour of Day',
                               labels={'hour_label': 'Hour of Day', 'comment_count': 'Number of Comments'},
                               height=400)
            
            # Add markers for better visibility
            fig_hourly.update_traces(mode='lines+markers', marker=dict(size=6))
            
            # Highlight peak hour
            peak_hour_idx = hourly_data['comment_count'].idxmax()
            peak_hour_label = hourly_data.loc[peak_hour_idx, 'hour_label']
            peak_count = hourly_data.loc[peak_hour_idx, 'comment_count']
            
            # Add annotation for peak hour
            fig_hourly.add_annotation(
                x=peak_hour_label,
                y=peak_count,
                text=f"Peak: {peak_hour_label}<br>{peak_count:,} comments",
                showarrow=True,
                arrowhead=2,
                arrowcolor="red",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="red",
                borderwidth=1
            )
        else:
            st.warning("⚠️ Hourly patterns CSV not found. Cannot display hourly chart.")
            return
        
    except Exception as e:
        st.error(f"❌ Error loading hourly patterns: {str(e)}")
        return
    
    apply_brand_style(fig_hourly)
    st.plotly_chart(fig_hourly, use_container_width=True)
    
    # Weekday patterns
    st.markdown("#### 📅 Weekday Activity Patterns")
    
    try:
        if weekday_data is not None and not weekday_data.empty:
            # Create bar chart using weekday and comment_count
            fig_weekday = px.bar(weekday_data, x='weekday', y='comment_count',
                                title='Comments by Day of Week',
                                labels={'weekday': 'Day of Week', 'comment_count': 'Number of Comments'},
                                height=400)
            
            # Highlight peak day
            peak_day_idx = weekday_data['comment_count'].idxmax()
            peak_day = weekday_data.loc[peak_day_idx, 'weekday']
            
            # Color bars - highlight peak day in red
            colors = ['#ED1B2E' if day == peak_day else '#1f77b4' for day in weekday_data['weekday']]
            fig_weekday.update_traces(marker_color=colors)
        else:
            st.warning("⚠️ Weekday patterns CSV not found. Cannot display weekday chart.")
            return
        
    except Exception as e:
        st.error(f"❌ Error loading weekday patterns: {str(e)}")
        return
    
    apply_brand_style(fig_weekday)
    st.plotly_chart(fig_weekday, use_container_width=True)
    
    # Moving averages
    st.markdown("#### 📊 Moving Averages")
    
    window_size = st.slider("Moving average window (days):", 3, 30, 7)
    
    # Calculate moving average using the already loaded daily_counts
    if daily_counts is not None:
        daily_counts['moving_avg'] = daily_counts['comment_count'].rolling(window=window_size, center=True).mean()
    else:
        st.error("❌ No daily data available for moving averages.")
        return
    
    fig_ma = go.Figure()
    
    # Add actual data
    fig_ma.add_trace(go.Scatter(
        x=daily_counts['date'], 
        y=daily_counts['comment_count'],
        mode='lines',
        name='Daily Count',
        line=dict(color='lightblue', width=1),
        opacity=0.7
    ))
    
    # Add moving average
    fig_ma.add_trace(go.Scatter(
        x=daily_counts['date'], 
        y=daily_counts['moving_avg'],
        mode='lines',
        name=f'{window_size}-day Moving Average',
        line=dict(color='#ED1B2E', width=3)
    ))
    
    fig_ma.update_layout(
        title=f'Daily Comment Count with {window_size}-Day Moving Average',
        xaxis_title='Date',
        yaxis_title='Number of Comments',
        hovermode='x unified',
        height=450,  # Fixed height to prevent stretching
        margin=dict(l=40, r=40, t=60, b=40)  # Proper margins
    )
    
    apply_brand_style(fig_ma)
    st.plotly_chart(fig_ma, use_container_width=True)