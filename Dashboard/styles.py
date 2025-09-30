"""
L'Oréal Dashboard CSS Styles
Professional styling for the TrendSpotter dashboard
"""

def get_loreal_styles():
    """Return the complete L'Oréal branded CSS styles for the dashboard"""
    return """
<style>
    /* L'Oréal Brand Colors */
    :root {
        --loreal-black: #000000;
        --loreal-white: #FFFFFF;
        --loreal-red: #ED1B2E;
        --loreal-grey: #6D6E70;
        --loreal-light-grey: #D7D7D8;
    }

    /* Main app styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
        background-color: #FFFFFF;
    }

    .main {
        background-color: #FFFFFF !important;
    }

    .stApp {
        background-color: #FFFFFF !important;
    }

    /* Header styling */
    .main h1 {
        color: var(--loreal-black) !important;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        text-align: center;
    }

    .main h2 {
        color: var(--loreal-black) !important;
        font-weight: 600;
        font-size: 1.8rem;
        margin-bottom: 1rem;
    }

    .main h3 {
        color: var(--loreal-black) !important;
        font-weight: 600;
        font-size: 1.4rem;
        margin-bottom: 0.75rem;
    }

    .main h4 {
        color: var(--loreal-black) !important;
        font-weight: 600;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }

    /* General text / paragraphs / divs */
    .main p, .main div {
        color: var(--loreal-black) !important;
    }
    
    /* Force ALL text elements to be black */
    .main * {
        color: var(--loreal-black) !important;
    }
    
    /* Override any specific text color classes */
    .main span, .main label, .main small, .main strong, .main em {
        color: var(--loreal-black) !important;
    }
    
    /* Force all subheaders to be black - Enhanced Rules */
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {
        color: var(--loreal-black) !important;
    }
    
    /* Force Streamlit subheaders to be black */
    .main .stSubheader, .main .stSubheader * {
        color: var(--loreal-black) !important;
    }
    
    /* Force any remaining header elements */
    .main [data-testid="stSubheader"], .main [data-testid="stSubheader"] * {
        color: var(--loreal-black) !important;
    }
    
    /* Additional subheader overrides */
    .main .element-container h3, .main .element-container h4, .main .element-container h5, .main .element-container h6 {
        color: var(--loreal-black) !important;
    }
    
    /* Force any text that might be styled as grey */
    .main .stSubheader, .main .stMarkdown h3, .main .stMarkdown h4, .main .stMarkdown h5, .main .stMarkdown h6 {
        color: var(--loreal-black) !important;
    }
    
    /* Enhanced Streamlit subheader targeting */
    div[data-testid="stSubheader"] h3,
    div[data-testid="stSubheader"] h4,
    div[data-testid="stSubheader"] h5,
    div[data-testid="stSubheader"] h6,
    div[data-testid="stSubheader"] * {
        color: var(--loreal-black) !important;
    }
    
    /* Target Streamlit's internal subheader classes */
    .main .stSubheader > div,
    .main .stSubheader > div > *,
    .main .element-container .stSubheader,
    .main .element-container .stSubheader * {
        color: var(--loreal-black) !important;
    }
    
    /* Nuclear option for subheaders - target all possible Streamlit subheader selectors */
    .main [class*="subheader"],
    .main [class*="subheader"] *,
    .main [class*="Subheader"],
    .main [class*="Subheader"] * {
        color: var(--loreal-black) !important;
    }
    
    /* AGGRESSIVE SUBHEADER TARGETING - Force all possible subheader elements */
    .stSubheader,
    .stSubheader *,
    [data-testid="stSubheader"],
    [data-testid="stSubheader"] *,
    .element-container .stSubheader,
    .element-container .stSubheader *,
    .stMarkdown .stSubheader,
    .stMarkdown .stSubheader *,
    .block-container .stSubheader,
    .block-container .stSubheader * {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    
    /* Target specific Streamlit subheader DOM structure */
    div[data-testid="element-container"] div[data-testid="stSubheader"],
    div[data-testid="element-container"] div[data-testid="stSubheader"] *,
    div[data-testid="element-container"] .stSubheader,
    div[data-testid="element-container"] .stSubheader * {
        color: #000000 !important;
    }
    
    /* Force subheader text content specifically */
    .stSubheader h3,
    .stSubheader h4,
    .stSubheader h5,
    .stSubheader h6,
    [data-testid="stSubheader"] h3,
    [data-testid="stSubheader"] h4,
    [data-testid="stSubheader"] h5,
    [data-testid="stSubheader"] h6 {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    
    /* Ultimate nuclear option - override any grey colors */
    .main *[style*="color: rgb(49, 51, 63)"],
    .main *[style*="color: #31333f"],
    .main *[style*="color: grey"],
    .main *[style*="color: gray"] {
        color: #000000 !important;
    }
    
    /* Override Streamlit's default text colors */
    .main .stMarkdown, .main .stText, .main .stWrite {
        color: var(--loreal-black) !important;
    }
    
    /* Force metric text to be black */
    .main [data-testid="metric-container"] * {
        color: var(--loreal-black) !important;
    }
    
    /* Override any remaining text elements */
    .main .element-container, .main .block-container {
        color: var(--loreal-black) !important;
    }
    
    /* Nuclear option - force ALL text to be black */
    .main, .main *, .main *::before, .main *::after {
        color: var(--loreal-black) !important;
    }
    
    /* Specific overrides for common Streamlit elements */
    .main .stAlert, .main .stSuccess, .main .stWarning, .main .stError {
        color: var(--loreal-black) !important;
    }
    
    .main .stAlert *, .main .stSuccess *, .main .stWarning *, .main .stError * {
        color: var(--loreal-black) !important;
    }

    /* Metric container / styling */
    .main [data-testid="metric-container"] {
        background-color: var(--loreal-white) !important;
        border: 1px solid var(--loreal-light-grey) !important;
        padding: 1rem !important;
        border-radius: 8px !important;
    }

    /* Metric label and value overrides */
    .main [data-testid="metric-container"] [data-testid="metric-label"] {
        color: var(--loreal-black) !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }

    .main [data-testid="metric-container"] [data-testid="metric-value"] {
        color: var(--loreal-black) !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }

    /* Additional more specific overrides (for deeper inner structure) */
    div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
        color: var(--loreal-black) !important;
    }

    div[data-testid="metric-container"] > label[data-testid="stMetricValue"] > div {
        color: var(--loreal-black) !important;
    }

    /* Container / panels styling */
    .main [data-testid="stHorizontalBlock"] > div {
        border: 1px solid var(--loreal-light-grey) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        background-color: #F5F5F5 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: var(--loreal-white) !important;
        color: var(--loreal-grey) !important;
        border: 1px solid var(--loreal-light-grey) !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 500 !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--loreal-red) !important;
        color: var(--loreal-white) !important;
        border-color: var(--loreal-red) !important;
    }

    /* Sidebar */
    .css-1d391kg {
        background-color: #F5F5F5 !important;
    }

    .sidebar .sidebar-content {
        background-color: #F5F5F5 !important;
    }

    .sidebar h3 {
        color: var(--loreal-black) !important;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        background-color: var(--loreal-white) !important;
        border: 1px solid var(--loreal-light-grey) !important;
        color: var(--loreal-black) !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: var(--loreal-red) !important;
        color: var(--loreal-white) !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
    }

    .stButton > button:hover {
        background-color: #c41e3a !important;
    }

    /* Divider */
    .stDivider {
        border-color: var(--loreal-light-grey) !important;
    }

    /* Alerts / messages */
    .stSuccess {
        background-color: #f0f9ff !important;
        border-left: 4px solid var(--loreal-red) !important;
    }

    .stWarning {
        background-color: #fffbeb !important;
        border-left: 4px solid #f59e0b !important;
    }

    .stError {
        background-color: #fef2f2 !important;
        border-left: 4px solid var(--loreal-red) !important;
    }

    /* Charts (Plotly) */
    .js-plotly-plot {
        background-color: var(--loreal-white) !important;
        border: 1px solid var(--loreal-light-grey) !important;
        border-radius: 8px !important;
    }

    /* Form / input labels */
    .stSelectbox label, .stMultiselect label, .stSlider label {
        color: var(--loreal-black) !important;
        font-weight: 600 !important;
    }

    .stSelectbox > div > div > div {
        color: var(--loreal-black) !important;
    }

    .stMultiselect > div > div {
        background-color: var(--loreal-white) !important;
        border: 1px solid var(--loreal-light-grey) !important;
        color: var(--loreal-black) !important;
    }

    .stSlider > div > div > div {
        background-color: var(--loreal-red) !important;
    }

    .stSlider > div > div > div > div {
        background-color: var(--loreal-red) !important;
    }

    /* Bordered containers */
    .stContainer {
        border: 1px solid var(--loreal-light-grey) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        background-color: #F5F5F5 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    }

    /* DataFrame / tables */
    .stDataFrame {
        border: 1px solid var(--loreal-light-grey) !important;
        border-radius: 8px !important;
    }
    
    /* COMPREHENSIVE TABLE STYLING - White background, black text */
    
    /* Target all Streamlit dataframes and tables */
    .stDataFrame,
    .stDataFrame *,
    [data-testid="stDataFrame"],
    [data-testid="stDataFrame"] *,
    .dataframe,
    .dataframe *,
    table,
    table * {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }
    
    /* Table headers */
    .stDataFrame thead,
    .stDataFrame thead *,
    .stDataFrame th,
    .stDataFrame th *,
    [data-testid="stDataFrame"] thead,
    [data-testid="stDataFrame"] thead *,
    [data-testid="stDataFrame"] th,
    [data-testid="stDataFrame"] th *,
    table thead,
    table thead *,
    table th,
    table th * {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 1px solid #cccccc !important;
        font-weight: 600 !important;
    }
    
    /* Table body and cells */
    .stDataFrame tbody,
    .stDataFrame tbody *,
    .stDataFrame td,
    .stDataFrame td *,
    [data-testid="stDataFrame"] tbody,
    [data-testid="stDataFrame"] tbody *,
    [data-testid="stDataFrame"] td,
    [data-testid="stDataFrame"] td *,
    table tbody,
    table tbody *,
    table td,
    table td * {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 1px solid #cccccc !important;
    }
    
    /* Table rows */
    .stDataFrame tr,
    .stDataFrame tr *,
    [data-testid="stDataFrame"] tr,
    [data-testid="stDataFrame"] tr *,
    table tr,
    table tr * {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }
    
    /* Hover states - light gray background but keep black text */
    .stDataFrame tr:hover,
    .stDataFrame tr:hover *,
    [data-testid="stDataFrame"] tr:hover,
    [data-testid="stDataFrame"] tr:hover *,
    table tr:hover,
    table tr:hover * {
        background-color: #f5f5f5 !important;
        color: #000000 !important;
    }
    
    /* Index column styling */
    .stDataFrame .index_name,
    .stDataFrame .index_name *,
    [data-testid="stDataFrame"] .index_name,
    [data-testid="stDataFrame"] .index_name * {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        font-weight: 600 !important;
    }
    
    /* Streamlit specific table elements */
    div[data-testid="stDataFrame"] div,
    div[data-testid="stDataFrame"] div *,
    .element-container .stDataFrame,
    .element-container .stDataFrame *,
    .block-container .stDataFrame,
    .block-container .stDataFrame * {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }
    
    /* Override any dark theme table styles */
    .stDataFrame [class*="dark"],
    .stDataFrame [class*="Dark"],
    [data-testid="stDataFrame"] [class*="dark"],
    [data-testid="stDataFrame"] [class*="Dark"] {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }
    
    /* Force table container backgrounds */
    .stDataFrame > div,
    .stDataFrame > div > div,
    [data-testid="stDataFrame"] > div,
    [data-testid="stDataFrame"] > div > div {
        background-color: #FFFFFF !important;
    }
    
    /* Pandas DataFrame specific styling */
    .dataframe thead tr th,
    .dataframe tbody tr td,
    .dataframe tbody tr th {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 1px solid #cccccc !important;
        text-align: left !important;
    }
    
    /* Alternative row styling (zebra striping) */
    .stDataFrame tbody tr:nth-child(even),
    [data-testid="stDataFrame"] tbody tr:nth-child(even),
    table tbody tr:nth-child(even) {
        background-color: #f9f9f9 !important;
        color: #000000 !important;
    }
    
    .stDataFrame tbody tr:nth-child(odd),
    [data-testid="stDataFrame"] tbody tr:nth-child(odd),
    table tbody tr:nth-child(odd) {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }

    /* Word cloud / custom containers */
    .wordcloud-container {
        background-color: #F5F5F5 !important;
        border: 1px solid var(--loreal-light-grey) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    }

    /* Alert boxes styling */
    .stAlert {
        border-left: 4px solid var(--loreal-red) !important;
        background-color: #fef2f2 !important;
    }

    /* Progress bars */
    .stProgress > div > div > div > div {
        background-color: var(--loreal-red) !important;
    }

    /* Expander header / content */
    .streamlit-expanderHeader {
        background-color: #F5F5F5 !important;
        color: var(--loreal-black) !important;
        border: 1px solid var(--loreal-light-grey) !important;
    }

    .streamlit-expanderContent {
        background-color: #F5F5F5 !important;
        border: 1px solid var(--loreal-light-grey) !important;
        border-top: none !important;
    }

    /* Code block styling */
    .stCode {
        background-color: #f8f9fa !important;
        border: 1px solid var(--loreal-light-grey) !important;
        border-radius: 6px !important;
    }

    /* Metric card styling */
    .metric-card {
        border: 2px solid black !important;
        border-radius: 12px !important;
        padding: 20px 15px !important;
        margin: 8px 0 !important;
        background-color: white !important;
        text-align: center !important;
    }

    /* Force all table text to be black */
    .main table, .main table *, .main table td, .main table th {
        color: var(--loreal-black) !important;
    }

    /* DataFrame styling */
    .main [data-testid="stDataFrame"] table,
    .main [data-testid="stDataFrame"] table *,
    .main [data-testid="stDataFrame"] table td,
    .main [data-testid="stDataFrame"] table th {
        color: var(--loreal-black) !important;
        background-color: white !important;
    }

    /* Force all data in tables to be black */
    .main .dataframe, .main .dataframe * {
        color: var(--loreal-black) !important;
    }

    /* Table headers and cells */
    .main thead, .main tbody, .main tr, .main td, .main th {
        color: var(--loreal-black) !important;
        background-color: white !important;
    }

    /* COMPREHENSIVE TABLE TEXT OVERRIDE - Force ALL text to be black */
    .main [data-testid="stDataFrame"] * {
        color: var(--loreal-black) !important;
        background-color: white !important;
    }

    .main .stDataFrame,
    .main .stDataFrame *,
    .main .stDataFrame table,
    .main .stDataFrame table *,
    .main .stDataFrame table td,
    .main .stDataFrame table th,
    .main .stDataFrame table tr,
    .main .stDataFrame table tbody,
    .main .stDataFrame table thead {
        color: var(--loreal-black) !important;
        background-color: white !important;
    }

    /* Force all text elements in tables to be black */
    .main table span, .main table div, .main table p, .main table strong, .main table em {
        color: var(--loreal-black) !important;
    }

    /* Nuclear option for all table-related elements */
    .main [data-testid="stDataFrame"] *,
    .main .dataframe *,
    .main table * {
        color: var(--loreal-black) !important;
    }

    /* Additional overrides for Streamlit dataframe components */
    .main [data-testid="stDataFrame"] div[data-testid="stDataFrame"],
    .main [data-testid="stDataFrame"] div[data-testid="stDataFrame"] * {
        color: var(--loreal-black) !important;
    }

    /* MINIMALIST BLACK AND WHITE TABLE STYLING */
    .main .stDataFrame {
        border-radius: 4px !important;
        overflow: hidden !important;
        box-shadow: none !important;
        border: 1px solid #cccccc !important;
    }

    /* Minimalist header styling - clean and modern */
    .main .stDataFrame th {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        text-transform: none !important;
        letter-spacing: normal !important;
        padding: 12px 8px !important;
        border-bottom: 1px solid #cccccc !important;
        border-left: none !important;
        border-right: none !important;
        text-align: left !important;
    }

    /* Minimalist cell styling - clean black text on white */
    .main .stDataFrame td {
        color: #000000 !important;
        background-color: #FFFFFF !important;
        padding: 10px 8px !important;
        border-bottom: 1px solid #cccccc !important;
        border-left: none !important;
        border-right: none !important;
        font-size: 14px !important;
        font-weight: normal !important;
    }

    /* Clean alternating rows - subtle grey */
    .main .stDataFrame tbody tr:nth-child(even) td {
        background-color: #f9f9f9 !important;
    }

    .main .stDataFrame tbody tr:hover td {
        background-color: #f0f0f0 !important;
        transition: background-color 0.2s ease !important;
    }

    /* Minimalist table styling for all tables */
    .main table {
        border-radius: 4px !important;
        overflow: hidden !important;
        box-shadow: none !important;
        border: 1px solid #cccccc !important;
    }

    .main table th {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        text-transform: none !important;
        letter-spacing: normal !important;
        padding: 12px 8px !important;
        border-bottom: 1px solid #cccccc !important;
        border-left: none !important;
        border-right: none !important;
        text-align: left !important;
    }

    .main table td {
        color: #000000 !important;
        background-color: #FFFFFF !important;
        padding: 10px 8px !important;
        border-bottom: 1px solid #cccccc !important;
        border-left: none !important;
        border-right: none !important;
        font-size: 14px !important;
        font-weight: normal !important;
    }

    .main table tbody tr:nth-child(even) td {
        background-color: #f9f9f9 !important;
    }

    .main table tbody tr:hover td {
        background-color: #f0f0f0 !important;
        transition: background-color 0.2s ease !important;
    }

    /* Minimalist styling for Streamlit dataframes */
    .main [data-testid="stDataFrame"] {
        border-radius: 4px !important;
        overflow: hidden !important;
        box-shadow: none !important;
        border: 1px solid #cccccc !important;
    }

    .main [data-testid="stDataFrame"] th {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        text-transform: none !important;
        letter-spacing: normal !important;
        padding: 12px 8px !important;
        border-bottom: 1px solid #cccccc !important;
        border-left: none !important;
        border-right: none !important;
    }

    .main [data-testid="stDataFrame"] td {
        color: #000000 !important;
        background-color: #FFFFFF !important;
        padding: 10px 8px !important;
        border-bottom: 1px solid #cccccc !important;
        font-size: 14px !important;
        font-weight: normal !important;
    }

    /* Professional scrollbar styling */
    .main .stDataFrame::-webkit-scrollbar {
        height: 8px !important;
    }

    .main .stDataFrame::-webkit-scrollbar-track {
        background: #F3F4F6 !important;
        border-radius: 4px !important;
    }

    .main .stDataFrame::-webkit-scrollbar-thumb {
        background: #ED1B2E !important;
        border-radius: 4px !important;
    }

    .main .stDataFrame::-webkit-scrollbar-thumb:hover {
        background: #C41E3A !important;
    }

    /* COMPREHENSIVE TABLES STYLING FOR ALL TABS */
    /* Ensure ALL table types across ALL tabs use L'Oréal branding */
    
    /* Minimalist dataframe tables styling */
    .main .dataframe,
    .main .dataframe table {
        border-radius: 4px !important;
        overflow: hidden !important;
        box-shadow: none !important;
        border: 1px solid #cccccc !important;
    }
    
    .main .dataframe th {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        text-transform: none !important;
        letter-spacing: normal !important;
        padding: 12px 8px !important;
        border-bottom: 1px solid #cccccc !important;
        border-left: none !important;
        border-right: none !important;
    }
    
    .main .dataframe td {
        color: #000000 !important;
        background-color: #FFFFFF !important;
        padding: 10px 8px !important;
        border-bottom: 1px solid #cccccc !important;
        font-size: 14px !important;
        font-weight: normal !important;
    }
    
    .main .dataframe tbody tr:nth-child(even) td {
        background-color: #f9f9f9 !important;
    }
    
    .main .dataframe tbody tr:hover td {
        background-color: #f0f0f0 !important;
        transition: background-color 0.2s ease !important;
    }

    /* Category analysis tables */
    .main [data-testid="stDataFrame"] table,
    .main [data-testid="stDataFrame"] table th,
    .main [data-testid="stDataFrame"] table td {
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
    }

    /* Classification model tables */
    .main .element-container table,
    .main .element-container table th,
    .main .element-container table td {
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
    }

    .main .element-container table th {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        text-transform: none !important;
        letter-spacing: normal !important;
        padding: 12px 8px !important;
        border-bottom: 1px solid #cccccc !important;
        border-left: none !important;
        border-right: none !important;
    }

    .main .element-container table td {
        color: #000000 !important;
        background-color: #FFFFFF !important;
        padding: 10px 8px !important;
        border-bottom: 1px solid #cccccc !important;
        font-size: 14px !important;
        font-weight: normal !important;
    }

    /* Spam detection tables */
    .main .block-container table,
    .main .block-container table th,
    .main .block-container table td {
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
    }

    /* Universal table styling for any remaining tables */
    .main div table,
    .main div table th,
    .main div table td {
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
    }

    .main div table th {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        text-transform: none !important;
        letter-spacing: normal !important;
        padding: 12px 8px !important;
        border-bottom: 1px solid #cccccc !important;
        border-left: none !important;
        border-right: none !important;
    }

    .main div table td {
        color: #000000 !important;
        background-color: #FFFFFF !important;
        padding: 10px 8px !important;
        border-bottom: 1px solid #cccccc !important;
        font-size: 14px !important;
        font-weight: normal !important;
    }

    .main div table tbody tr:nth-child(even) td {
        background-color: #f9f9f9 !important;
    }

    .main div table tbody tr:hover td {
        background-color: #f0f0f0 !important;
        transition: background-color 0.2s ease !important;
    }

    /* Number input styling */
    .main .stNumberInput label {
        color: var(--loreal-black) !important;
        font-weight: 600 !important;
    }

    .main .stNumberInput input {
        background-color: white !important;
        color: var(--loreal-black) !important;
        border: 1px solid var(--loreal-light-grey) !important;
    }

    .main .stNumberInput input:focus {
        background-color: white !important;
        color: var(--loreal-black) !important;
        border: 2px solid var(--loreal-red) !important;
        box-shadow: 0 0 0 1px var(--loreal-red) !important;
    }

    /* L'Oréal Header Styling */
    .loreal-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 2rem;
        background-color: var(--loreal-white);
        border-bottom: 2px solid var(--loreal-red);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        width: 100%;
        box-sizing: border-box;
    }

    .loreal-logo {
        height: 50px;
        width: auto;
        max-width: 150px;
        object-fit: contain;
    }

    .loreal-title-section {
        text-align: center;
        flex-grow: 1;
        padding: 0 1rem;
    }

    .loreal-title-section h1 {
        color: var(--loreal-black) !important;
        font-family: Arial, sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 0.25rem 0;
        line-height: 1.2;
    }

    .loreal-title-section p {
        color: var(--loreal-grey) !important;
        font-family: Arial, sans-serif;
        font-size: 1rem;
        font-weight: 500;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .loreal-spacer {
        width: 150px;
        flex-shrink: 0;
    }

    /* Responsive design for header */
    @media (max-width: 1024px) {
        .loreal-header {
            padding: 0.75rem 1.5rem;
        }
        
        .loreal-title-section h1 {
            font-size: 1.8rem;
        }
        
        .loreal-title-section p {
            font-size: 0.9rem;
        }
        
        .loreal-logo {
            height: 40px;
        }
        
        .loreal-spacer {
            width: 100px;
        }
    }

    @media (max-width: 768px) {
        .loreal-header {
            flex-direction: column;
            text-align: center;
            padding: 1rem;
            gap: 0.75rem;
        }
        
        .loreal-title-section {
            padding: 0;
        }
        
        .loreal-title-section h1 {
            font-size: 1.6rem;
        }
        
        .loreal-title-section p {
            font-size: 0.85rem;
        }
        
        .loreal-logo {
            height: 35px;
        }
        
        .loreal-spacer {
            display: none;
        }
    }

    @media (max-width: 480px) {
        .loreal-header {
            padding: 0.75rem;
        }
        
        .loreal-title-section h1 {
            font-size: 1.4rem;
        }
        
        .loreal-title-section p {
            font-size: 0.8rem;
        }
        
        .loreal-logo {
            height: 30px;
        }
    }

        /* 🔹 Force ALL metric labels (top text) to black */
    div[data-testid="stMetric"] label[data-testid="stMetricLabel"] p,
    div[data-testid="stMetric"] label[data-testid="stMetricLabel"] * {
        color: var(--loreal-black) !important;
    }

    /* 🔹 Force ALL metric values (numbers) to black */
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] * {
        color: var(--loreal-black) !important;
    }

</style>
"""