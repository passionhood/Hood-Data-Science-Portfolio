# Passion Hood
# MDSC 20009: Machine Learning Application Project

# Importing Libraries

# Core data handling
import os
import numpy as np
import pandas as pd

# Streamlit (web app framework)
import streamlit as st

# Visualization libraries
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

# Sidebar navigation
from streamlit_option_menu import option_menu

# Machine learning tools
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)

# Page Configuration 

st.set_page_config(
    page_title="Investment Risk Analyzer",
    layout="wide",  # Use full screen width
)

# File path setup 

# Get the directory of this script
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to sample dataset
SAMPLE_FILE_PATH = os.path.join(APP_DIR, "30_yr_market_data_with_formulas.xlsx")

# User Interface Styling, with custom CSS to improve layout and sidebar appearance

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    section[data-testid="stSidebar"] {
        background-color: #f5f6fa;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Constants,expected columns in the dataset and mapping price columns to return columns for feature engineering

# Key financial assets expected in dataset
BASE_ASSET_COLUMNS = [
    "Nasdaq", "NYSE Composite", "S&P500", "CBOE Volitility",
    "Amazon", "Apple", "Exxon", "Google", "Tesla",
    "Crude Oil-WTI", "Gold",
]

# Mapping price columns to return columns
RETURN_MAP = {
    "Nasdaq": "nasdaq_return",
    "NYSE Composite": "nyse_return",
    "S&P500": "sp500_return",
}

# Data loading functions, with caching to improve performance and a special function to handle Excel files with non-standard headers

@st.cache_data
def load_excel_with_detected_header(file):
    """
    Load Excel file and automatically detect header row.
    This fixes issues when the first row is not actual column names.
    """
    preview = pd.read_excel(file, header=None, nrows=10)

    header_row = 0
    for i in range(len(preview)):
        if "date" in [str(x).lower() for x in preview.iloc[i]]:
            header_row = i
            break

    return pd.read_excel(file, header=header_row)


@st.cache_data
def load_uploaded_data(file):
    """
    Load user-uploaded dataset (CSV or Excel).
    """
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return load_excel_with_detected_header(file)

# Data cleaning function, to standardize column names, convert date formats, and ensure numeric columns are properly typed

def clean_dataframe(df):
    """
    Clean dataset:
    - Fix column names
    - Convert Date column
    - Convert numeric columns
    """
    df = df.copy()

    # Strip spaces from column names
    df.columns = [str(col).strip() for col in df.columns]

    # Standardize Date column
    for col in df.columns:
        if col.lower() == "date":
            df = df.rename(columns={col: "Date"})

    # Convert Date column
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Convert other columns to numeric
    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# Feature engineering function, to create ML-ready features such as daily returns, rolling volatility, and a risk classification label based on volatility thresholds

def engineer_features(df):
    """
    Create ML-ready features:
    - Daily returns
    - Rolling volatility
    - Risk classification
    """
    df = df.copy()

    # Compute returns
    if "S&P500" in df.columns:
        df["sp500_return"] = df["S&P500"].pct_change()

    # Rolling volatility
    df["sp500_volatility_5d"] = df["sp500_return"].rolling(5).std()

    # Risk label (high volatility = 1)
    threshold = df["sp500_volatility_5d"].quantile(0.7)
    df["risk"] = (df["sp500_volatility_5d"] > threshold).astype(int)

    return df

# Model building function, to create a machine learning pipeline that includes preprocessing steps (imputation and scaling) followed by the selected model (Logistic Regression or Decision Tree)

def build_model(name, params):
    """
    Create ML pipeline with preprocessing + model.
    """
    if name == "Logistic Regression":
        return Pipeline([
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(**params))
        ])

    if name == "Decision Tree":
        return Pipeline([
            ("imputer", SimpleImputer()),
            ("model", DecisionTreeClassifier(**params))
        ])

# Sidebar navigation, allowing users to switch between Dashboard, Train Model, and Predict pages, as well as upload their dataset or choose to use the sample dataset

with st.sidebar:
    page = option_menu(
        "Navigation",
        ["Dashboard", "Train Model", "Predict"],
        icons=["house", "sliders", "search"]
    )

    uploaded_file = st.file_uploader("Upload dataset")

    use_sample = st.checkbox("Use sample dataset")

# Loading data, with priority given to the sample dataset if the checkbox is selected, otherwise loading the user-uploaded file. The loaded data is then cleaned and features are engineered to prepare it for analysis and modeling.

df = None

if use_sample and os.path.exists(SAMPLE_FILE_PATH):
    df = load_excel_with_detected_header(SAMPLE_FILE_PATH)
elif uploaded_file:
    df = load_uploaded_data(uploaded_file)

if df is not None:
    df = clean_dataframe(df)
    df = engineer_features(df)

# App title for the application, displayed at the top of the page

st.title("Investment Risk Analyzer")

# Dashboard page, which displays a line chart of the selected asset's price over time. It checks if the dataset is loaded and contains a Date column before allowing the user to select an asset and visualize its price history.

if page == "Dashboard":
    st.header("Dashboard")

    if df is None:
        st.warning("Upload data first.")
    elif "Date" not in df.columns:
        st.error("Dataset must include a Date column.")
    else:
        asset = st.selectbox("Select asset", BASE_ASSET_COLUMNS)

        st.plotly_chart(
            px.line(df, x="Date", y=asset, title=f"{asset} Price Over Time")
        )

# Model training page, which allows users to select features, choose a model type (Logistic Regression or Decision Tree), and set hyperparameters. It then trains the model on the selected features and target variable, evaluates its performance using accuracy and a confusion matrix, and stores the trained model and feature list in the session state for later use in predictions.

elif page == "Train Model":
    st.header("Train Model")

    if df is None:
        st.warning("Upload data first.")
    else:
        features = st.multiselect(
            "Features",
            [col for col in df.columns if col != "risk"]
        )

        target = "risk"

        model_name = st.selectbox("Model", ["Logistic Regression", "Decision Tree"])

        params = {}

        if model_name == "Logistic Regression":
            params["max_iter"] = st.slider("Max Iter", 100, 1000, 200)

        if model_name == "Decision Tree":
            params["max_depth"] = st.slider("Max Depth", 1, 10, 3)

        if st.button("Train Model"):

            # Split data into training and testing sets
            X = df[features].dropna()
            y = df.loc[X.index, target]

            X_train, X_test, y_train, y_test = train_test_split(X, y)

            # Train model
            model = build_model(model_name, params)
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)

            # Metrics
            acc = accuracy_score(y_test, y_pred)

            st.metric("Accuracy", acc)

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            st.write(cm)

            st.session_state["model"] = model
            st.session_state["features"] = features

# Prediction page, which allows users to input values for the selected features and uses the trained model to predict the risk classification (high or low). It checks if a model has been trained and if the necessary features are available before allowing the user to make predictions.

elif page == "Predict":
    st.header("Predict")

    if "model" not in st.session_state:
        st.warning("Train a model first.")
    else:
        model = st.session_state["model"]
        features = st.session_state["features"]

        inputs = {}
        for f in features:
            inputs[f] = st.number_input(f)

        if st.button("Predict"):
            pred = model.predict(pd.DataFrame([inputs]))[0]

            if pred == 1:
                st.error("High Risk")
            else:
                st.success("Low Risk")