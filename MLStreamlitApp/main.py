# Investment Risk Analyzer - Machine Learning Streamlit Application
# Author: Passion Hood
# Course: MDSC 20009: Machine Learning for Data Science
# Description: Interactive web application for financial data analysis and
#              supervised machine learning model training with real-time
#              hyperparameter tuning and risk prediction.

# Imports and dependencies for data handling, visualization, machine learning, and Streamlit components
import os
import numpy as np
import pandas as pd

# Streamlit framework for interactive web applications
import streamlit as st

# Visualization libraries for charts and plots
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

# Navigation menu component for Streamlit
from streamlit_option_menu import option_menu

# scikit-learn: machine learning library
from sklearn.model_selection import train_test_split       # Data splitting
from sklearn.pipeline import Pipeline                      # Model pipelines
from sklearn.impute import SimpleImputer                   # Missing value handling
from sklearn.preprocessing import StandardScaler           # Feature scaling
from sklearn.linear_model import LogisticRegression        # Linear model
from sklearn.tree import DecisionTreeClassifier            # Tree-based model
from sklearn.neighbors import KNeighborsClassifier         # Distance-based model

# Evaluation metrics for model performance assessment
from sklearn.metrics import (
    accuracy_score,      # Correct predictions / total predictions
    precision_score,     # True positives / (true positives + false positives)
    recall_score,        # True positives / (true positives + false negatives)
    f1_score,            # Harmonic mean of precision and recall
    confusion_matrix,    # Matrix of prediction outcomes
    roc_curve,           # Receiver Operating Characteristic curve
    auc,                 # Area Under the Curve
)

# Page configuration and custom CSS styling for Streamlit components to enhance UI appearance and user experience

# Configure Streamlit page settings
st.set_page_config(
    page_title="Investment Risk Analyzer",  # Browser tab title
    layout="wide",                          # Use full screen width
)

# Application directory and sample dataset path configuration

# Get the absolute path to the directory containing this script
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Define path to the sample dataset (30-year historical market data)
SAMPLE_FILE_PATH = os.path.join(APP_DIR, "30_yr_market_data_with_formulas.xlsx")

# Custom CSS styling for Streamlit components to enhance UI appearance and user experience

st.markdown(
    """
    <style>
    /* Main container: limit width and add padding */
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Sidebar styling: light background color */
    section[data-testid="stSidebar"] {
        background-color: #f5f6fa;
    }

    /* Metric cards: add borders and rounded corners */
    div[data-testid="stMetric"] {
        border: 1px solid rgba(49, 51, 63, 0.10);
        border-radius: 14px;
        padding: 12px 16px;
        background-color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Constants and configurations for data processing, feature engineering, and model building

# List of financial assets expected in the dataset
# Includes major indices, tech stocks, and commodities
BASE_ASSET_COLUMNS = [
    "Nasdaq",              # NASDAQ Composite Index
    "NYSE Composite",      # New York Stock Exchange Composite
    "S&P500",              # S&P 500 Index
    "CBOE Volitility",     # CBOE Volatility Index (VIX)
    "Amazon",              # Amazon stock price
    "Apple",               # Apple stock price
    "Exxon",               # Exxon Mobil stock price
    "Google",              # Google stock price
    "Tesla",               # Tesla stock price
    "Crude Oil-WTI",       # West Texas Intermediate crude oil price
    "Gold",                # Gold price
]

# Mapping from asset price columns to their corresponding return columns
# Used to quickly find the return series for each asset
RETURN_MAP = {
    "Nasdaq": "nasdaq_return",
    "NYSE Composite": "nyse_return",
    "S&P500": "sp500_return",
    "Amazon": "amazon_return",
    "Apple": "apple_return",
    "Exxon": "exxon_return",
    "Google": "google_return",
    "Tesla": "tesla_return",
    "Crude Oil-WTI": "oil_return",
    "Gold": "gold_return",
}

# Helper function to read the sample dataset file as bytes for download functionality, with caching to optimize performance

@st.cache_data
def get_sample_file_bytes(file_path: str) -> bytes:
    """
    Read the sample dataset file as bytes for download functionality.
    
    Purpose:
        Enable users to download the sample dataset template from the app
        without re-reading the file on every app interaction.
    
    Parameters:
        file_path (str): Absolute path to the sample dataset file
    
    Returns:
        bytes: File contents as binary data for download
    
    Note:
        Decorated with @st.cache_data for performance optimization
    """
    with open(file_path, "rb") as f:
        return f.read()


@st.cache_data
def load_excel_with_detected_header(file_source):
    """
    Load an Excel file and automatically detect the header row.
    
    Purpose:
        Handle Excel files where the header row is not in the first row,
        which is common in real-world financial data with metadata.
    
    Parameters:
        file_source: File path (str) or file object from st.file_uploader()
    
    Returns:
        pd.DataFrame: Loaded data with correctly identified headers
    
    Logic:
        1. Read first 10 rows without headers to scan for header row
        2. Search each row for the word "date" (case-insensitive)
        3. Use the row containing "date" as the header
        4. Re-read file with correct header row index
    
    Note:
        Decorated with @st.cache_data for performance optimization
    """
    # Preview first 10 rows without assuming any header
    preview = pd.read_excel(file_source, sheet_name=0, header=None, nrows=10)

    # Default to row 0 if no "date" column is found
    header_row = 0
    
    # Scan each row to find the header row containing "date"
    for i in range(len(preview)):
        # Convert row values to lowercase strings for case-insensitive comparison
        row_values = [str(x).strip().lower() for x in preview.iloc[i].tolist()]
        if "date" in row_values:
            header_row = i
            break

    # Load entire file with detected header row
    df = pd.read_excel(file_source, sheet_name=0, header=header_row)
    return df


@st.cache_data
def load_uploaded_data(uploaded_file):
    """
    Load CSV or Excel data from user-uploaded file.
    
    Purpose:
        Provide flexible file upload support for both CSV and Excel formats,
        automatically selecting the appropriate loading function.
    
    Parameters:
        uploaded_file: File object from st.file_uploader() or None
    
    Returns:
        pd.DataFrame: Loaded data if file is valid, None otherwise
    
    Supported Formats:
        - .csv: Comma-separated values
        - .xlsx: Excel 2007+ format
        - .xls: Excel 97-2003 format
    """
    if uploaded_file is None:
        return None

    # Normalize filename to lowercase for extension checking
    file_name = uploaded_file.name.lower()

    # Load CSV files
    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    # Load Excel files (with automatic header detection)
    if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        return load_excel_with_detected_header(uploaded_file)

    # Return None if file format is unsupported
    return None

# Helper function to clean and standardize the dataset for machine learning, including fixing column names, converting data types, handling dates, and sorting chronologically

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize the dataset for machine learning.
    
    Purpose:
        Prepare raw data by fixing column names, converting data types,
        handling dates, and sorting chronologically.
    
    Parameters:
        df (pd.DataFrame): Input dataframe to clean
    
    Returns:
        pd.DataFrame: Cleaned dataframe with standardized columns and types
    
    Processing Steps:
        1. Strip whitespace from column names
        2. Normalize any date-like column header to "Date"
        3. Convert Date column to datetime format
        4. Convert all non-Date columns to numeric (coerce errors to NaN)
        5. Sort by Date in ascending order
    
    Note:
        Creates a copy to avoid modifying the original dataframe
    """
    df = df.copy()
    
    # Step 1: Remove leading/trailing whitespace from column names
    df.columns = [str(col).strip() for col in df.columns]

    # Step 2: Find and standardize date column name
    # This handles variations like "date", "Date", "DATE", etc.
    for col in df.columns:
        if str(col).strip().lower() == "date":
            df = df.rename(columns={col: "Date"})

    # Step 3: Convert Date column to datetime format
    # errors="coerce" converts unparseable values to NaT (Not a Time)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Step 4: Convert all non-Date columns to numeric
    # This ensures compatibility with machine learning algorithms
    # errors="coerce" converts non-numeric values to NaN
    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Step 5: Sort by Date in ascending order and reset index
    if "Date" in df.columns:
        df = df.sort_values("Date").reset_index(drop=True)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create machine learning features from raw price data.
    
    Purpose:
        Transform raw asset prices into ML-ready features including returns,
        volatility measures, and risk labels for supervised learning.
    
    Parameters:
        df (pd.DataFrame): Cleaned dataframe with price columns
    
    Returns:
        pd.DataFrame: Original dataframe plus engineered features
    
    Features Created:
        - Daily Returns: Percentage price change from previous day
        - Rolling Volatility: 5-day standard deviation of returns
        - Risk Label: Binary target (1=High Risk, 0=Low Risk)
    
    Feature Engineering Logic:
        1. Calculate daily returns for each asset using percent change
        2. Calculate 5-day rolling volatility of S&P 500 returns
        3. Define high risk as volatility above the 70th percentile
        4. Create binary risk label for supervised learning
    
    Note:
        Creates a copy to avoid modifying the original dataframe
    """
    df = df.copy()

    # Calculate daily returns for Nasdaq index
    if "Nasdaq" in df.columns:
        df["nasdaq_return"] = df["Nasdaq"].pct_change()

    # Calculate daily returns for NYSE Composite index
    if "NYSE Composite" in df.columns:
        df["nyse_return"] = df["NYSE Composite"].pct_change()

    # Calculate daily returns for S&P 500 index
    if "S&P500" in df.columns:
        df["sp500_return"] = df["S&P500"].pct_change()

    # Calculate daily returns for Amazon stock
    if "Amazon" in df.columns:
        df["amazon_return"] = df["Amazon"].pct_change()

    # Calculate daily returns for Apple stock
    if "Apple" in df.columns:
        df["apple_return"] = df["Apple"].pct_change()

    # Calculate daily returns for Exxon stock
    if "Exxon" in df.columns:
        df["exxon_return"] = df["Exxon"].pct_change()

    # Calculate daily returns for Google stock
    if "Google" in df.columns:
        df["google_return"] = df["Google"].pct_change()

    # Calculate daily returns for Tesla stock
    if "Tesla" in df.columns:
        df["tesla_return"] = df["Tesla"].pct_change()

    # Calculate daily returns for Crude Oil (WTI)
    if "Crude Oil-WTI" in df.columns:
        df["oil_return"] = df["Crude Oil-WTI"].pct_change()

    # Calculate daily returns for Gold
    if "Gold" in df.columns:
        df["gold_return"] = df["Gold"].pct_change()

    # Create volatility and risk features (only if S&P 500 returns exist)
    if "sp500_return" in df.columns:
        # Calculate 5-day rolling standard deviation of S&P 500 returns
        # This measures market volatility over a 5-day window
        df["sp500_volatility_5d"] = df["sp500_return"].rolling(window=5).std()
        
        # Define high risk threshold as 70th percentile of volatility
        # Days above this threshold are classified as high risk
        threshold = df["sp500_volatility_5d"].quantile(0.70)
        
        # Create binary risk label: 1 for high risk, 0 for low risk
        # This will be the target variable for supervised learning
        df["risk"] = (df["sp500_volatility_5d"] > threshold).astype(int)

    return df

# Helper function to extract numeric columns with non-null values for model training and analysis

def get_numeric_columns(df: pd.DataFrame) -> list:
    """
    Extract numeric columns from dataframe that contain at least one non-null value.
    
    Purpose:
        Identify columns suitable for machine learning features and analysis.
        Filters out columns that are entirely empty.
    
    Parameters:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        list: Names of numeric columns with at least one non-null value
    """
    # Select columns with numeric data types (int, float)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter to keep only columns with at least one non-null value
    # This removes completely empty columns
    return [col for col in numeric_cols if df[col].notna().sum() > 0]


def safe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare dataframe for Streamlit display to avoid PyArrow conversion errors.
    
    Purpose:
        Convert problematic object columns to strings to prevent rendering
        errors in Streamlit table displays.
    
    Parameters:
        df (pd.DataFrame): Input dataframe with mixed data types
    
    Returns:
        pd.DataFrame: Dataframe with object columns converted to strings
    
    Note:
        Fills NaN values with empty strings before conversion
    """
    display_df = df.copy()

    # Process each column
    for col in display_df.columns:
        # Check if column contains mixed object types
        if display_df[col].dtype == "object":
            # Fill null values and convert to string
            display_df[col] = display_df[col].fillna("").astype(str)

    return display_df

# Helper function to build machine learning pipelines based on user-selected model and hyperparameters

def build_model(model_name: str, params: dict):
    """
    Construct an ML pipeline with preprocessing and model.
    
    Purpose:
        Create consistent, reproducible ML pipelines that handle missing values,
        feature scaling, and model training in a single object.
    
    Parameters:
        model_name (str): Type of model - "Logistic Regression", "Decision Tree",
                         or "K-Nearest Neighbors"
        params (dict): Model-specific hyperparameters
    
    Returns:
        sklearn.pipeline.Pipeline: Complete preprocessing + model pipeline
    
    Pipelines:
        1. Logistic Regression:
           - Input: Raw features with missing values
           - Step 1: SimpleImputer (fill NaN with median)
           - Step 2: StandardScaler (normalize features)
           - Step 3: LogisticRegression (binary classification)
           - Output: Probability predictions
        
        2. Decision Tree:
           - Input: Raw features with missing values
           - Step 1: SimpleImputer (fill NaN with median)
           - Step 2: DecisionTreeClassifier (tree-based classification)
           - Output: Class predictions
           - Note: No scaling (trees are scale-invariant)
        
        3. K-Nearest Neighbors:
           - Input: Raw features with missing values
           - Step 1: SimpleImputer (fill NaN with median)
           - Step 2: StandardScaler (normalize features)
           - Step 3: KNeighborsClassifier (distance-based classification)
           - Output: Class predictions
    
    Raises:
        ValueError: If model_name is not recognized
    """
    
    # Logistic Regression: Linear model with scaling
    if model_name == "Logistic Regression":
        return Pipeline(
            [
                # Fill missing values with median (robust to outliers)
                ("imputer", SimpleImputer(strategy="median")),
                
                # Scale features to have mean=0 and std=1
                # Required for logistic regression convergence
                ("scaler", StandardScaler()),
                
                # Logistic regression classifier with hyperparameters
                (
                    "model",
                    LogisticRegression(
                        C=params["C"],                    # Inverse regularization strength
                        max_iter=params["max_iter"],      # Max iterations for convergence
                        random_state=42,                  # Reproducible results
                    ),
                ),
            ]
        )

    # Decision Tree: Tree-based model without scaling
    if model_name == "Decision Tree":
        return Pipeline(
            [
                # Fill missing values with median
                ("imputer", SimpleImputer(strategy="median")),
                
                # Decision tree classifier with hyperparameters
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=params["max_depth"],                      # Max tree depth
                        min_samples_split=params["min_samples_split"],      # Min samples to split
                        random_state=42,                                    # Reproducible results
                    ),
                ),
            ]
        )

    # K-Nearest Neighbors: Distance-based model with scaling
    if model_name == "K-Nearest Neighbors":
        return Pipeline(
            [
                # Fill missing values with median
                ("imputer", SimpleImputer(strategy="median")),
                
                # Scale features (required for distance-based models)
                ("scaler", StandardScaler()),
                
                # K-Nearest Neighbors classifier with hyperparameter
                ("model", KNeighborsClassifier(n_neighbors=params["n_neighbors"])),
            ]
        )

    # Raise error for unsupported models
    raise ValueError("Unsupported model selected.")

# Helper functions for visualizations and data retrieval for dashboard metrics and plots

def make_line_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, y_title: str):
    """
    Create an interactive line chart using Plotly.
    
    Purpose:
        Generate reusable, interactive time-series visualizations
        for exploring asset prices, returns, and volatility.
    
    Parameters:
        df (pd.DataFrame): Data containing the columns to plot
        x_col (str): Column name for x-axis (typically "Date")
        y_col (str): Column name for y-axis (asset price, returns, volatility)
        title (str): Chart title
        y_title (str): Y-axis label
    
    Returns:
        plotly.graph_objects.Figure: Interactive line chart
    
    Features:
        - Removes missing values before plotting
        - White background template for clean appearance
        - Fixed height (420px) for consistent layout
        - Minimal margins for compact display
    """
    # Remove rows with missing values in either column
    chart_df = df[[x_col, y_col]].dropna()

    # Create figure object
    fig = go.Figure()
    
    # Add line trace to plot
    fig.add_trace(
        go.Scatter(
            x=chart_df[x_col],      # X-axis data (dates)
            y=chart_df[y_col],      # Y-axis data (prices/returns/volatility)
            mode="lines",           # Display as connected line
            name=y_col,             # Legend name
        )
    )
    
    # Update layout for appearance
    fig.update_layout(
        title=title,                           # Chart title
        xaxis_title=x_col,                     # X-axis label
        yaxis_title=y_title,                   # Y-axis label
        template="plotly_white",               # White background
        margin=dict(l=20, r=20, t=50, b=20),  # Minimal margins
        height=420,                            # Fixed height
    )
    
    return fig


def make_confusion_matrix_plot(cm: np.ndarray, class_names: list):
    """
    Create a confusion matrix visualization using Matplotlib.
    
    Purpose:
        Display prediction accuracy breakdown across all classes,
        showing true positives, true negatives, false positives, false negatives.
    
    Parameters:
        cm (np.ndarray): Confusion matrix from sklearn.metrics.confusion_matrix
        class_names (list): List of class names/labels for display
    
    Returns:
        matplotlib.figure.Figure: Confusion matrix heatmap
    
    Matrix Interpretation:
        - Diagonal (top-left to bottom-right): Correct predictions
        - Off-diagonal: Misclassifications
        - Cell [i,j]: Predictions of class j when actual class was i
    """
    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Display confusion matrix as heatmap (colors indicate values)
    ax.imshow(cm, interpolation="nearest")
    
    # Set title and axis labels
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    
    # Set tick positions and labels for classes
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    # Add text annotations with count values in each cell
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return fig


def make_roc_plot(y_true, y_proba):
    """
    Create ROC (Receiver Operating Characteristic) curve visualization.
    
    Purpose:
        Display model's classification performance across all thresholds,
        showing the trade-off between true positive rate and false positive rate.
    
    Parameters:
        y_true: True binary class labels (0 or 1)
        y_proba: Predicted probabilities for the positive class (0-1)
    
    Returns:
        tuple: (matplotlib.figure.Figure, float)
            - Figure: ROC curve plot
            - roc_auc: Area under the curve (0-1, higher is better)
    
    Interpretation:
        - Diagonal line (0,0 to 1,1): Random classifier (AUC = 0.5)
        - Curve above diagonal: Good classifier (AUC > 0.5)
        - Curve below diagonal: Poor classifier (AUC < 0.5)
        - Top-left curve: Perfect classifier (AUC = 1.0)
    """
    # Calculate ROC curve: false positive rate, true positive rate, thresholds
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    
    # Calculate area under the ROC curve (AUC)
    # AUC = probability that model ranks random positive instance higher than random negative
    roc_auc = auc(fpr, tpr)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot ROC curve with AUC label
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    
    # Plot diagonal reference line (random classifier)
    ax.plot([0, 1], [0, 1], linestyle="--")
    
    # Set labels and title
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    
    # Adjust layout
    plt.tight_layout()

    return fig, roc_auc


def get_selected_rows(df: pd.DataFrame, selected_date):
    """
    Retrieve current and previous row data for a selected date.
    
    Purpose:
        Extract data for calculating daily changes and percentage returns
        for the selected date in the Dashboard.
    
    Parameters:
        df (pd.DataFrame): Dataframe with Date column
        selected_date: Selected date from date picker
    
    Returns:
        tuple: (current_row, previous_row) as pandas Series
            - Returns (None, None) if date not found or Date column missing
    
    Logic:
        1. Check if Date column exists
        2. Find rows matching selected_date
        3. Return current row and previous row for comparison
    """
    # Validate Date column exists
    if "Date" not in df.columns:
        return None, None

    # Convert selected_date to datetime and find matching rows
    row_df = df[df["Date"] == pd.to_datetime(selected_date)]
    
    # Return None if no matching date found
    if row_df.empty:
        return None, None

    # Get index of the matching row
    idx = row_df.index[0]
    
    # Get current row data
    current_row = df.loc[idx]
    
    # Get previous row if it exists, otherwise use current row
    previous_row = df.loc[idx - 1] if idx > 0 else current_row
    
    return current_row, previous_row

# Sidebar navagation and data upload section: Allows users to navigate between pages and upload datasets for analysis and modeling

with st.sidebar:
    # Sidebar heading
    st.markdown("## Menu")

    # Navigation menu with 4 pages
    page = option_menu(
        "Navigation",  # Menu title
        ["Dashboard", "Explore Data", "Train Model", "Predict"],  # Page options
        icons=["house", "bar-chart", "sliders", "search"],        # Icons for each page
        menu_icon="list",                                           # Menu icon
        default_index=0,                                            # Default to Dashboard
        # Custom styling for menu appearance
        styles={
            "container": {
                "padding": "0.8rem",
                "background-color": "#ffffff",
                "border-radius": "14px",
            },
            "icon": {
                "color": "#2f2f3a",
                "font-size": "22px",
            },
            "nav-link": {
                "font-size": "19px",
                "text-align": "left",
                "margin": "8px 0px",
                "padding": "14px 18px",
                "border-radius": "10px",
                "--hover-color": "#f4f4f4",
            },
            "nav-link-selected": {
                "background-color": "#ff4b4b",
                "color": "white",
                "font-weight": "600",
            },
        },
    )

    # Separator line in sidebar
    st.markdown("---")
    
    # Dataset section header
    st.subheader("Dataset")

    # File upload widget for CSV or Excel files
    uploaded_file = st.file_uploader(
        "Upload a CSV or Excel file",
        type=["csv", "xlsx", "xls"],  # Accepted file types
    )

    # Checkbox to use sample dataset instead
    use_sample_data = st.checkbox("Use sample dataset")

    # Download button for sample dataset (if it exists)
    if os.path.exists(SAMPLE_FILE_PATH):
        st.download_button(
            label="Download sample dataset",
            data=get_sample_file_bytes(SAMPLE_FILE_PATH),  # File contents as bytes
            file_name="30_yr_market_data_with_formulas.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.caption("Sample dataset not found in this folder.")

# Data loading logic based on user input (sample dataset takes priority over uploaded file)

df = None  # Initialize dataframe as None

# If user selected sample dataset and file exists
if use_sample_data and os.path.exists(SAMPLE_FILE_PATH):
    df = load_excel_with_detected_header(SAMPLE_FILE_PATH)  # Load sample file
    df = clean_dataframe(df)                                  # Clean data
    df = engineer_features(df)                                # Create features

# Otherwise, if user uploaded a file
elif uploaded_file is not None:
    df = load_uploaded_data(uploaded_file)  # Load uploaded file
    if df is not None:
        df = clean_dataframe(df)             # Clean data
        df = engineer_features(df)           # Create features

# App header and description

# Main app title
st.title("Investment Risk Analyzer")

# Subtitle describing app functionality
st.write(
    "Explore financial market data, view investment metrics by date, and train a supervised machine learning model to classify investment risk."
)

# Dashboard page: display key metrics and visualizations for selected asset and date

if page == "Dashboard":
    st.header("Dashboard")

    # Check if data is loaded
    if df is None:
        st.warning("Please upload a dataset or use the sample dataset.")
    
    # Check if Date column exists
    elif "Date" not in df.columns:
        st.error("Your dataset must include a Date column.")
    
    # Data is loaded and valid
    else:
        # Filter available assets (only those in the uploaded dataset)
        asset_options = [col for col in BASE_ASSET_COLUMNS if col in df.columns]

        if not asset_options:
            st.error("No expected asset columns were found in the dataset.")
        else:
            # Create two columns for asset selection and date selection
            c1, c2 = st.columns(2)

            # Column 1: Asset selection dropdown
            with c1:
                # Default to S&P500 if available, otherwise first asset
                default_asset_index = asset_options.index("S&P500") if "S&P500" in asset_options else 0
                asset_col = st.selectbox("Select asset", asset_options, index=default_asset_index)

            # Column 2: Date selection picker
            with c2:
                # Get valid dates from dataset (sorted, unique, no NaT)
                valid_dates = df["Date"].dropna().sort_values().unique()
                selected_date = st.date_input(
                    "Select date",
                    value=pd.to_datetime(valid_dates[-1]).date(),  # Default to latest date
                    min_value=pd.to_datetime(valid_dates[0]).date(),
                    max_value=pd.to_datetime(valid_dates[-1]).date(),
                )

            # Retrieve data for selected date and previous date
            current_row, previous_row = get_selected_rows(df, selected_date)

            if current_row is None:
                st.warning("No data found for the selected date.")
            else:
                # Extract values from current and previous rows
                current_value = current_row.get(asset_col, np.nan)
                previous_value = previous_row.get(asset_col, np.nan)

                # Calculate daily change (absolute)
                delta_value = (
                    current_value - previous_value
                    if pd.notna(current_value) and pd.notna(previous_value)
                    else np.nan
                )
                
                # Calculate daily change (percentage)
                delta_pct = (
                    (delta_value / previous_value) * 100
                    if pd.notna(delta_value) and pd.notna(previous_value) and previous_value != 0
                    else np.nan
                )

                # Get asset return (if available) and risk label
                return_col = RETURN_MAP.get(asset_col)
                return_value = current_row.get(return_col, np.nan) if return_col in df.columns else np.nan
                risk_value = current_row.get("risk", np.nan)

                # Display 4 key metric cards
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric(
                        f"{asset_col} Value",
                        f"{current_value:.4f}" if pd.notna(current_value) else "N/A"
                    )
                with m2:
                    st.metric(
                        "Daily % Change",
                        f"{delta_pct:.2f}%" if pd.notna(delta_pct) else "N/A"
                    )
                with m3:
                    st.metric(
                        "Asset Return",
                        f"{return_value:.4f}" if pd.notna(return_value) else "N/A"
                    )
                with m4:
                    if pd.notna(risk_value):
                        st.metric("Risk Label", "High Risk" if int(risk_value) == 1 else "Low Risk")
                    else:
                        st.metric("Risk Label", "N/A")

                # Create 4 tabs for different visualizations
                tab1, tab2, tab3, tab4 = st.tabs(
                    ["Price", "Returns", "Volatility", "Risk Distribution"]
                )

                # Tab 1: Asset price over time
                with tab1:
                    st.plotly_chart(
                        make_line_chart(df, "Date", asset_col, f"{asset_col} Over Time", asset_col),
                        use_container_width=True,
                    )

                # Tab 2: Asset returns over time
                with tab2:
                    if return_col in df.columns:
                        st.plotly_chart(
                            make_line_chart(df, "Date", return_col, f"{asset_col} Daily Returns", "Return"),
                            use_container_width=True,
                        )
                    else:
                        st.info("Return series not available for this asset.")

                # Tab 3: S&P 500 volatility over time
                with tab3:
                    if "sp500_volatility_5d" in df.columns:
                        st.plotly_chart(
                            make_line_chart(
                                df,
                                "Date",
                                "sp500_volatility_5d",
                                "S&P500 5-Day Volatility",
                                "Volatility",
                            ),
                            use_container_width=True,
                        )
                    else:
                        st.info("Volatility data not available.")

                # Tab 4: Distribution of risk labels
                with tab4:
                    if "risk" in df.columns:
                        # Count observations in each risk category
                        risk_counts = df["risk"].value_counts().sort_index()

                        # Create bar chart
                        fig = go.Figure()
                        fig.add_bar(
                            x=["Low Risk", "High Risk"],
                            y=[risk_counts.get(0, 0), risk_counts.get(1, 0)],
                        )
                        fig.update_layout(
                            title="Risk Label Distribution",
                            xaxis_title="Risk Category",
                            yaxis_title="Count",
                            template="plotly_white",
                            height=420,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Risk labels not available.")

# Explore Data page: Provides dataset preview, summary statistics, time-series visualization, and correlation matrix analysis

elif page == "Explore Data":
    st.header("Explore Data")

    if df is None:
        st.warning("Please upload a dataset or use the sample dataset.")
    else:
        # Get all numeric columns for analysis
        numeric_cols = get_numeric_columns(df)

        # Create two-column layout
        left, right = st.columns(2)

        # Left column: Dataset preview
        with left:
            st.subheader("Dataset preview")
            st.dataframe(safe_for_display(df.head(20)), use_container_width=True)

        # Right column: Summary statistics
        with right:
            st.subheader("Summary statistics")
            if numeric_cols:
                # Calculate and display statistics
                summary_df = df[numeric_cols].describe().transpose()
                st.dataframe(safe_for_display(summary_df), use_container_width=True)
            else:
                st.info("No numeric columns available for summary statistics.")

        # Separator line between sections
        st.markdown("---")

        # Time-series visualization section
        if "Date" in df.columns and numeric_cols:
            # Dropdown to select column for visualization
            graph_col = st.selectbox("Choose a numeric column to graph", numeric_cols)
            st.plotly_chart(
                make_line_chart(df, "Date", graph_col, f"{graph_col} Over Time", graph_col),
                use_container_width=True,
            )

        # Separator line before correlation matrix
        st.markdown("---")

        # Correlation matrix section (requires at least 2 numeric columns)
        if len(numeric_cols) >= 2:
            st.subheader("Correlation Matrix")
            
            # Calculate correlation between all numeric columns
            corr = df[numeric_cols].corr()

            # Create heatmap with Plotly Express
            # RdBu_r colorscale: Red=positive correlation, Blue=negative correlation
            fig = px.imshow(
                corr,
                text_auto=".2f",         # Display correlation values
                aspect="auto",            # Auto aspect ratio
                color_continuous_scale="RdBu_r",  # Red-Blue diverging scale
            )
            fig.update_layout(height=700)  # Large height for readability
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("At least two numeric columns are needed for a correlation matrix.")

# Train Model page: Provides interface for selecting features, model type, hyperparameters, and training the model

elif page == "Train Model":
    st.header("Train Model")

    if df is None:
        st.warning("Please upload a dataset or use the sample dataset.")
    else:
        # Get numeric columns for model training
        numeric_cols = get_numeric_columns(df)

        if len(numeric_cols) < 2:
            st.error("You need at least two numeric columns to train a model.")
        else:
            # Target column selection
            default_target = "risk" if "risk" in numeric_cols else numeric_cols[-1]

            target_col = st.selectbox(
                "Target column",
                options=numeric_cols,
                index=numeric_cols.index(default_target),
            )

            # Feature columns (all numeric except target)
            feature_options = [col for col in numeric_cols if col != target_col]

            # Pre-select common feature columns for convenience
            default_features = [
                col for col in [
                    "nasdaq_return",
                    "nyse_return",
                    "sp500_return",
                    "amazon_return",
                    "apple_return",
                    "exxon_return",
                    "google_return",
                    "tesla_return",
                    "oil_return",
                    "gold_return",
                    "CBOE Volitility",
                    "sp500_volatility_5d",
                ]
                if col in feature_options
            ]

            # Multi-select for features
            selected_features = st.multiselect(
                "Feature columns",
                options=feature_options,
                default=default_features if default_features else feature_options[:5],
            )

            if not selected_features:
                st.warning("Please select at least one feature column.")
            else:
                # Create control panel: Model selection, test size, train button
                c1, c2, c3 = st.columns(3)

                with c1:
                    model_name = st.selectbox(
                        "Model",
                        ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors"],
                    )

                with c2:
                    # Slider for test set size (10-40%)
                    test_size = st.slider("Test size", 0.10, 0.40, 0.20, 0.05)

                with c3:
                    # Train button with full width
                    st.write("")
                    st.write("")
                    train_button = st.button("Train Model", use_container_width=True)

                # Hyperparameter selection based on model type
                params = {}

                if model_name == "Logistic Regression":
                    # Logistic Regression hyperparameters
                    p1, p2 = st.columns(2)
                    with p1:
                        # C: Inverse of regularization strength (lower = more regularization)
                        params["C"] = st.slider("C", 0.01, 10.0, 1.0, 0.01)
                    with p2:
                        # Max iterations for convergence
                        params["max_iter"] = st.slider("Max iterations", 100, 2000, 500, 100)

                elif model_name == "Decision Tree":
                    # Decision Tree hyperparameters
                    p1, p2 = st.columns(2)
                    with p1:
                        # Maximum tree depth
                        params["max_depth"] = st.slider("Max depth", 1, 20, 5)
                    with p2:
                        # Minimum samples required to split a node
                        params["min_samples_split"] = st.slider("Min samples split", 2, 20, 2)

                elif model_name == "K-Nearest Neighbors":
                    # K-Nearest Neighbors hyperparameter
                    params["n_neighbors"] = st.slider("Number of neighbors", 1, 25, 5)

                # Prepare data: drop rows with any missing values
                model_df = df[selected_features + [target_col]].dropna()

                X = model_df[selected_features]  # Features
                y = model_df[target_col]          # Target

                # Get unique classes in target variable
                unique_classes = sorted(y.unique())

                if len(unique_classes) < 2:
                    st.error("The target column must contain at least two classes.")
                elif train_button:
                    # Split data into train and test sets
                    # stratify ensures equal class distribution in train/test
                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        test_size=test_size,   # Proportion for test set
                        random_state=42,       # Reproducible split
                        stratify=y,            # Maintain class distribution
                    )

                    # Build and train model
                    model = build_model(model_name, params)
                    model.fit(X_train, y_train)
                    
                    # Make predictions on test set
                    y_pred = model.predict(X_test)

                    # Calculate performance metrics
                    acc = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                    # Display results
                    st.subheader("Model Performance")

                    # Display 4 performance metrics in cards
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Accuracy", f"{acc:.3f}")
                    with m2:
                        st.metric("Precision", f"{precision:.3f}")
                    with m3:
                        st.metric("Recall", f"{recall:.3f}")
                    with m4:
                        st.metric("F1 Score", f"{f1:.3f}")

                    # Display confusion matrix and ROC curve
                    left_plot, right_plot = st.columns(2)

                    with left_plot:
                        # Create and display confusion matrix
                        cm = confusion_matrix(y_test, y_pred)
                        fig_cm = make_confusion_matrix_plot(cm, [str(c) for c in unique_classes])
                        st.pyplot(fig_cm)

                    with right_plot:
                        # Create and display ROC curve (only for binary classification)
                        if len(unique_classes) == 2 and hasattr(model, "predict_proba"):
                            # Get probability predictions for positive class
                            y_proba = model.predict_proba(X_test)[:, 1]
                            fig_roc, roc_auc = make_roc_plot(y_test, y_proba)
                            st.pyplot(fig_roc)
                            st.metric("ROC-AUC", f"{roc_auc:.3f}")
                        else:
                            st.info("ROC curve is available only for binary classification.")

                    # Save trained model to session state for use in Predict page
                    st.session_state["trained_model"] = model
                    st.session_state["selected_features"] = selected_features
                    st.session_state["class_labels"] = unique_classes

                    # Success message
                    st.success("Model trained successfully.")

#  Predict page: Input feature values and get risk predictions from the trained model

elif page == "Predict":
    st.header("Predict")

    # Check if a model has been trained
    if "trained_model" not in st.session_state:
        st.warning("Train a model first in the Train Model page.")
    else:
        # Retrieve trained model and features from session state
        model = st.session_state["trained_model"]
        selected_features = st.session_state["selected_features"]
        class_labels = st.session_state["class_labels"]

        st.write("Enter values for a single observation:")

        # Dictionary to store user inputs
        input_data = {}
        
        # Create two-column layout for input fields
        cols = st.columns(2)

        # Generate input field for each feature
        for i, feature in enumerate(selected_features):
            with cols[i % 2]:  # Alternate between left and right column
                input_data[feature] = st.number_input(
                    feature,
                    value=0.0,
                    format="%.6f",  # 6 decimal places
                )

        # Convert user inputs to dataframe (required for model.predict)
        input_df = pd.DataFrame([input_data])

        if st.button("Generate Prediction"):
            # Make prediction
            prediction = model.predict(input_df)[0]

            # Display prediction result with color-coding
            if str(prediction) == "1" or prediction == 1:
                st.error("Predicted class: High Risk")
            elif str(prediction) == "0" or prediction == 0:
                st.success("Predicted class: Low Risk")
            else:
                st.info(f"Predicted class: {prediction}")

            # Display class probabilities (if model supports them)
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                st.subheader("Class Probabilities")

                # Display probability for each class
                for label, prob in zip(class_labels, probabilities):
                    st.write(f"Class {label}: {prob:.3f}")