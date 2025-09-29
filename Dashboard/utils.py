"""
Utility functions for the L'Oréal TrendSpotter Dashboard
Contains helper functions for logo encoding, data loading, and UI components
"""

import base64
import pandas as pd
import streamlit as st
import os
from pathlib import Path


def encode_logo_to_base64(image_path):
    """Encode the L'Oréal logo to base64 for embedding in HTML"""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return encoded_string
    except FileNotFoundError:
        st.warning(f"Logo file not found at {image_path}")
        return None


def create_header_html(logo_base64):
    """Create the professional L'Oréal header HTML with logo"""
    if logo_base64:
        return f"""
        <div class="loreal-header">
            <img src="data:image/jpeg;base64,{logo_base64}" class="loreal-logo" alt="L'Oréal Logo">
            <div class="loreal-title-section">
                <h1>TrendSpotter</h1>
                <p>AI-Powered Social Media Analytics Platform</p>
            </div>
            <div class="loreal-spacer"></div>
        </div>
        """
    else:
        return """
        <div class="loreal-header">
            <div class="loreal-spacer"></div>
            <div class="loreal-title-section">
                <h1>TrendSpotter</h1>
                <p>AI-Powered Social Media Analytics Platform</p>
            </div>
            <div class="loreal-spacer"></div>
        </div>
        """


def load_data_with_fallback(data_paths, data_name="data"):
    """
    Load data with multiple fallback paths
    
    Args:
        data_paths (list): List of potential file paths to try
        data_name (str): Name of the data for error messages
        
    Returns:
        pandas.DataFrame or None: Loaded data or None if all paths fail
    """
    for path in data_paths:
        try:
            if os.path.exists(path):
                if path.endswith('.csv'):
                    data = pd.read_csv(path)
                    st.success(f"✅ {data_name} loaded successfully from: {path}")
                    return data
                elif path.endswith('.json'):
                    data = pd.read_json(path)
                    st.success(f"✅ {data_name} loaded successfully from: {path}")
                    return data
        except Exception as e:
            st.warning(f"⚠️ Failed to load {data_name} from {path}: {str(e)}")
            continue
    
    st.error(f"❌ Could not load {data_name} from any of the specified paths")
    return None


def get_data_paths():
    """Get standard data file paths for the dashboard"""
    base_paths = {
        'clustered_comments': [
            "AI_Model/Clustering Model For Comment Sub Category/clustered_comments_reassigned.csv",
            "../AI_Model/Clustering Model For Comment Sub Category/clustered_comments_reassigned.csv",
            "clustered_comments_reassigned.csv"
        ],
        'cluster_txt': [
            "cluster.txt",
            "../cluster.txt",
            "Dashboard/cluster.txt"
        ]
    }
    return base_paths





def create_metric_card(title, value, description=""):
    """Create a professional metric card with L'Oréal styling"""
    return f"""
    <div class="metric-card">
        <h4 style="margin-bottom: 10px; color: #000000 !important;">{title}</h4>
        <h2 style="margin-bottom: 5px; color: #ED1B2E !important; font-weight: bold;">{value}</h2>
        <p style="margin: 0; color: #6D6E70 !important; font-size: 0.9rem;">{description}</p>
    </div>
    """


def safe_file_check(file_path):
    """Safely check if a file exists and is readable"""
    try:
        return os.path.exists(file_path) and os.path.isfile(file_path)
    except Exception:
        return False


def get_file_size_mb(file_path):
    """Get file size in MB"""
    try:
        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)
        return round(size_mb, 2)
    except Exception:
        return 0


def create_download_link(df, filename, link_text="Download CSV"):
    """Create a download link for a DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="color: #ED1B2E; text-decoration: none; font-weight: bold;">{link_text}</a>'
    return href


def format_large_number(num):
    """Format large numbers with appropriate suffixes (K, M, B)"""
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)


def create_status_badge(status, message=""):
    """Create a status badge with appropriate styling"""
    status_colors = {
        'success': '#10B981',
        'warning': '#F59E0B', 
        'error': '#EF4444',
        'info': '#3B82F6'
    }
    
    color = status_colors.get(status, '#6B7280')
    
    return f"""
    <div style="
        display: inline-block;
        padding: 4px 12px;
        background-color: {color};
        color: white;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px;
    ">
        {message}
    </div>
    """


def validate_dataframe(df, required_columns=None):
    """Validate that a DataFrame has the required structure"""
    if df is None or df.empty:
        return False, "DataFrame is empty or None"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"


def get_current_directory():
    """Get the current working directory"""
    return os.getcwd()


def list_files_in_directory(directory, extension=None):
    """List files in a directory, optionally filtered by extension"""
    try:
        files = []
        for file in os.listdir(directory):
            if extension:
                if file.endswith(extension):
                    files.append(file)
            else:
                files.append(file)
        return files
    except Exception as e:
        st.error(f"Error listing files in {directory}: {str(e)}")
        return []


def display_device_info():
    """Display current device information (CPU/GPU)"""
    try:
        import torch
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            device_info = f"🚀 **GPU Acceleration Active**: {device_name}"
            device_type = "GPU"
        else:
            device_info = "💻 **Running on CPU**"
            device_type = "CPU"
        
        st.info(device_info)
        return device_type
        
    except ImportError:
        device_info = "💻 **Running on CPU** (PyTorch not available)"
        device_type = "CPU"
        st.info(device_info)
        return device_type