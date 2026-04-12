
# Author: Passion Hood
# Course: MDSC 20009: Machine Learning for Data Science

# Import Libraries 
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

from streamlit_option_menu import option_menu

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)

# PAGE CONFIG
st.set_page_config(
    page_title="Investment Risk Analyzer",
    layout="wide",
)

# APP PATHS
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_FILE_PATH = os.path.join(APP_DIR, "30_yr_market_data_with_formulas.xlsx")

# STYLING
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

# CONSTANTS
BASE_ASSET_COLUMNS = [
    "Nasdaq",
    "NYSE Composite",
    "S&P500",
    "CBOE Volitility",
    "Amazon",
    "Apple",
    "Exxon",
    "Google",
    "Tesla",
    "Crude Oil-WTI",
    "Gold",
]

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

# HELPER FUNCTIONS
@st.cache_data
def get_sample_file_bytes(file_path: str) -> bytes:
    """Read the sample dataset as bytes for download."""
    with open(file_path, "rb") as f:
        return f.read()


@st.cache_data
def load_excel_with_detected_header(file_source):
    """
    Load an Excel file and detect the header row by looking
    for the row containing 'Date'.
    """
    preview = pd.read_excel(file_source, sheet_name=0, header=None, nrows=10)

    header_row = 0
    for i in range(len(preview)):
        row_values = [str(x).strip().lower() for x in preview.iloc[i].tolist()]
        if "date" in row_values:
            header_row = i
            break

    df = pd.read_excel(file_source, sheet_name=0, header=header_row)
    return df


@st.cache_data
def load_uploaded_data(uploaded_file):
    """Load CSV or Excel data."""
    if uploaded_file is None:
        return None

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        return load_excel_with_detected_header(uploaded_file)

    return None


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset:
    - strip column names
    - normalize any date-like column name to 'Date'
    - convert non-Date columns to numeric where possible
    - sort by Date
    """
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    # Normalize any date-like column header
    for col in df.columns:
        if str(col).strip().lower() == "date":
            df = df.rename(columns={col: "Date"})

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Date" in df.columns:
        df = df.sort_values("Date").reset_index(drop=True)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ML-ready features directly in Python.
    """
    df = df.copy()

    if "Nasdaq" in df.columns:
        df["nasdaq_return"] = df["Nasdaq"].pct_change()

    if "NYSE Composite" in df.columns:
        df["nyse_return"] = df["NYSE Composite"].pct_change()

    if "S&P500" in df.columns:
        df["sp500_return"] = df["S&P500"].pct_change()

    if "Amazon" in df.columns:
        df["amazon_return"] = df["Amazon"].pct_change()

    if "Apple" in df.columns:
        df["apple_return"] = df["Apple"].pct_change()

    if "Exxon" in df.columns:
        df["exxon_return"] = df["Exxon"].pct_change()

    if "Google" in df.columns:
        df["google_return"] = df["Google"].pct_change()

    if "Tesla" in df.columns:
        df["tesla_return"] = df["Tesla"].pct_change()

    if "Crude Oil-WTI" in df.columns:
        df["oil_return"] = df["Crude Oil-WTI"].pct_change()

    if "Gold" in df.columns:
        df["gold_return"] = df["Gold"].pct_change()

    if "sp500_return" in df.columns:
        df["sp500_volatility_5d"] = df["sp500_return"].rolling(window=5).std()
        threshold = df["sp500_volatility_5d"].quantile(0.70)
        df["risk"] = (df["sp500_volatility_5d"] > threshold).astype(int)

    return df


def get_numeric_columns(df: pd.DataFrame) -> list:
    """Return numeric columns with at least one non-null value."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if df[col].notna().sum() > 0]


def safe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make dataframe safe for Streamlit display by converting object
    columns to strings to avoid PyArrow conversion errors.
    """
    display_df = df.copy()

    for col in display_df.columns:
        if display_df[col].dtype == "object":
            display_df[col] = display_df[col].fillna("").astype(str)

    return display_df


def build_model(model_name: str, params: dict):
    """Build an ML pipeline."""
    if model_name == "Logistic Regression":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=params["C"],
                        max_iter=params["max_iter"],
                        random_state=42,
                    ),
                ),
            ]
        )

    if model_name == "Decision Tree":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=params["max_depth"],
                        min_samples_split=params["min_samples_split"],
                        random_state=42,
                    ),
                ),
            ]
        )

    if model_name == "K-Nearest Neighbors":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=params["n_neighbors"])),
            ]
        )

    raise ValueError("Unsupported model selected.")


def make_line_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, y_title: str):
    """Create a reusable line chart."""
    chart_df = df[[x_col, y_col]].dropna()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_df[x_col],
            y=chart_df[y_col],
            mode="lines",
            name=y_col,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_title,
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        height=420,
    )
    return fig


def make_confusion_matrix_plot(cm: np.ndarray, class_names: list):
    """Create confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    return fig


def make_roc_plot(y_true, y_proba):
    """Create ROC curve and return AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    plt.tight_layout()

    return fig, roc_auc


def get_selected_rows(df: pd.DataFrame, selected_date):
    """Get current and previous row for a selected date."""
    if "Date" not in df.columns:
        return None, None

    row_df = df[df["Date"] == pd.to_datetime(selected_date)]
    if row_df.empty:
        return None, None

    idx = row_df.index[0]
    current_row = df.loc[idx]
    previous_row = df.loc[idx - 1] if idx > 0 else current_row
    return current_row, previous_row

# SIDEBAR
with st.sidebar:
    st.markdown("## Menu")

    page = option_menu(
        "Navigation",
        ["Dashboard", "Explore Data", "Train Model", "Predict"],
        icons=["house", "bar-chart", "sliders", "search"],
        menu_icon="list",
        default_index=0,
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

    st.markdown("---")
    st.subheader("Dataset")

    uploaded_file = st.file_uploader(
        "Upload a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
    )

    use_sample_data = st.checkbox("Use sample dataset")

    if os.path.exists(SAMPLE_FILE_PATH):
        st.download_button(
            label="Download sample dataset",
            data=get_sample_file_bytes(SAMPLE_FILE_PATH),
            file_name="30_yr_market_data_with_formulas.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.caption("Sample dataset not found in this folder.")

# LOAD DATA
df = None

if use_sample_data and os.path.exists(SAMPLE_FILE_PATH):
    df = load_excel_with_detected_header(SAMPLE_FILE_PATH)
    df = clean_dataframe(df)
    df = engineer_features(df)

elif uploaded_file is not None:
    df = load_uploaded_data(uploaded_file)
    if df is not None:
        df = clean_dataframe(df)
        df = engineer_features(df)

# APP HEADER
st.title("Investment Risk Analyzer")
st.write(
    "Explore financial market data, view investment metrics by date, and train a supervised machine learning model to classify investment risk."
)

# DASHBOARD
if page == "Dashboard":
    st.header("Dashboard")

    if df is None:
        st.warning("Please upload a dataset or use the sample dataset.")
    elif "Date" not in df.columns:
        st.error("Your dataset must include a Date column.")
    else:
        asset_options = [col for col in BASE_ASSET_COLUMNS if col in df.columns]

        if not asset_options:
            st.error("No expected asset columns were found in the dataset.")
        else:
            c1, c2 = st.columns(2)

            with c1:
                default_asset_index = asset_options.index("S&P500") if "S&P500" in asset_options else 0
                asset_col = st.selectbox("Select asset", asset_options, index=default_asset_index)

            with c2:
                valid_dates = df["Date"].dropna().sort_values().unique()
                selected_date = st.date_input(
                    "Select date",
                    value=pd.to_datetime(valid_dates[-1]).date(),
                    min_value=pd.to_datetime(valid_dates[0]).date(),
                    max_value=pd.to_datetime(valid_dates[-1]).date(),
                )

            current_row, previous_row = get_selected_rows(df, selected_date)

            if current_row is None:
                st.warning("No data found for the selected date.")
            else:
                current_value = current_row.get(asset_col, np.nan)
                previous_value = previous_row.get(asset_col, np.nan)

                delta_value = (
                    current_value - previous_value
                    if pd.notna(current_value) and pd.notna(previous_value)
                    else np.nan
                )
                delta_pct = (
                    (delta_value / previous_value) * 100
                    if pd.notna(delta_value) and pd.notna(previous_value) and previous_value != 0
                    else np.nan
                )

                return_col = RETURN_MAP.get(asset_col)
                return_value = current_row.get(return_col, np.nan) if return_col in df.columns else np.nan
                risk_value = current_row.get("risk", np.nan)

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

                tab1, tab2, tab3, tab4 = st.tabs(
                    ["Price", "Returns", "Volatility", "Risk Distribution"]
                )

                with tab1:
                    st.plotly_chart(
                        make_line_chart(df, "Date", asset_col, f"{asset_col} Over Time", asset_col),
                        use_container_width=True,
                    )

                with tab2:
                    if return_col in df.columns:
                        st.plotly_chart(
                            make_line_chart(df, "Date", return_col, f"{asset_col} Daily Returns", "Return"),
                            use_container_width=True,
                        )
                    else:
                        st.info("Return series not available for this asset.")

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

                with tab4:
                    if "risk" in df.columns:
                        risk_counts = df["risk"].value_counts().sort_index()

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

# =========================================================
# EXPLORE DATA
# =========================================================
elif page == "Explore Data":
    st.header("Explore Data")

    if df is None:
        st.warning("Please upload a dataset or use the sample dataset.")
    else:
        numeric_cols = get_numeric_columns(df)

        left, right = st.columns(2)

        with left:
            st.subheader("Dataset preview")
            st.dataframe(safe_for_display(df.head(20)), use_container_width=True)

        with right:
            st.subheader("Summary statistics")
            if numeric_cols:
                summary_df = df[numeric_cols].describe().transpose()
                st.dataframe(safe_for_display(summary_df), use_container_width=True)
            else:
                st.info("No numeric columns available for summary statistics.")

        st.markdown("---")

        if "Date" in df.columns and numeric_cols:
            graph_col = st.selectbox("Choose a numeric column to graph", numeric_cols)
            st.plotly_chart(
                make_line_chart(df, "Date", graph_col, f"{graph_col} Over Time", graph_col),
                use_container_width=True,
            )

        st.markdown("---")

        if len(numeric_cols) >= 2:
            st.subheader("Correlation Matrix")
            corr = df[numeric_cols].corr()

            fig = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
            )
            fig.update_layout(height=700)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("At least two numeric columns are needed for a correlation matrix.")

# TRAIN MODEL
elif page == "Train Model":
    st.header("Train Model")

    if df is None:
        st.warning("Please upload a dataset or use the sample dataset.")
    else:
        numeric_cols = get_numeric_columns(df)

        if len(numeric_cols) < 2:
            st.error("You need at least two numeric columns to train a model.")
        else:
            default_target = "risk" if "risk" in numeric_cols else numeric_cols[-1]

            target_col = st.selectbox(
                "Target column",
                options=numeric_cols,
                index=numeric_cols.index(default_target),
            )

            feature_options = [col for col in numeric_cols if col != target_col]

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

            selected_features = st.multiselect(
                "Feature columns",
                options=feature_options,
                default=default_features if default_features else feature_options[:5],
            )

            if not selected_features:
                st.warning("Please select at least one feature column.")
            else:
                c1, c2, c3 = st.columns(3)

                with c1:
                    model_name = st.selectbox(
                        "Model",
                        ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors"],
                    )

                with c2:
                    test_size = st.slider("Test size", 0.10, 0.40, 0.20, 0.05)

                with c3:
                    st.write("")
                    st.write("")
                    train_button = st.button("Train Model", use_container_width=True)

                params = {}

                if model_name == "Logistic Regression":
                    p1, p2 = st.columns(2)
                    with p1:
                        params["C"] = st.slider("C", 0.01, 10.0, 1.0, 0.01)
                    with p2:
                        params["max_iter"] = st.slider("Max iterations", 100, 2000, 500, 100)

                elif model_name == "Decision Tree":
                    p1, p2 = st.columns(2)
                    with p1:
                        params["max_depth"] = st.slider("Max depth", 1, 20, 5)
                    with p2:
                        params["min_samples_split"] = st.slider("Min samples split", 2, 20, 2)

                elif model_name == "K-Nearest Neighbors":
                    params["n_neighbors"] = st.slider("Number of neighbors", 1, 25, 5)

                model_df = df[selected_features + [target_col]].dropna()

                X = model_df[selected_features]
                y = model_df[target_col]

                unique_classes = sorted(y.unique())

                if len(unique_classes) < 2:
                    st.error("The target column must contain at least two classes.")
                elif train_button:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        test_size=test_size,
                        random_state=42,
                        stratify=y,
                    )

                    model = build_model(model_name, params)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    acc = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                    st.subheader("Model Performance")

                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Accuracy", f"{acc:.3f}")
                    with m2:
                        st.metric("Precision", f"{precision:.3f}")
                    with m3:
                        st.metric("Recall", f"{recall:.3f}")
                    with m4:
                        st.metric("F1 Score", f"{f1:.3f}")

                    left_plot, right_plot = st.columns(2)

                    with left_plot:
                        cm = confusion_matrix(y_test, y_pred)
                        fig_cm = make_confusion_matrix_plot(cm, [str(c) for c in unique_classes])
                        st.pyplot(fig_cm)

                    with right_plot:
                        if len(unique_classes) == 2 and hasattr(model, "predict_proba"):
                            y_proba = model.predict_proba(X_test)[:, 1]
                            fig_roc, roc_auc = make_roc_plot(y_test, y_proba)
                            st.pyplot(fig_roc)
                            st.metric("ROC-AUC", f"{roc_auc:.3f}")
                        else:
                            st.info("ROC curve is available only for binary classification.")

                    st.session_state["trained_model"] = model
                    st.session_state["selected_features"] = selected_features
                    st.session_state["class_labels"] = unique_classes

                    st.success("Model trained successfully.")

# PREDICT
elif page == "Predict":
    st.header("Predict")

    if "trained_model" not in st.session_state:
        st.warning("Train a model first in the Train Model page.")
    else:
        model = st.session_state["trained_model"]
        selected_features = st.session_state["selected_features"]
        class_labels = st.session_state["class_labels"]

        st.write("Enter values for a single observation:")

        input_data = {}
        cols = st.columns(2)

        for i, feature in enumerate(selected_features):
            with cols[i % 2]:
                input_data[feature] = st.number_input(
                    feature,
                    value=0.0,
                    format="%.6f",
                )

        input_df = pd.DataFrame([input_data])

        if st.button("Generate Prediction"):
            prediction = model.predict(input_df)[0]

            if str(prediction) == "1" or prediction == 1:
                st.error("Predicted class: High Risk")
            elif str(prediction) == "0" or prediction == 0:
                st.success("Predicted class: Low Risk")
            else:
                st.info(f"Predicted class: {prediction}")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                st.subheader("Class Probabilities")

                for label, prob in zip(class_labels, probabilities):
                    st.write(f"Class {label}: {prob:.3f}")