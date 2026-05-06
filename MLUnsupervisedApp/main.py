# Investor Persona Clustering App
# Author: Passion Hood
# Course: MDSC 20009: Machine Learning for Data Science
#
# Project Purpose:
# This Streamlit application demonstrates unsupervised machine learning by
# identifying hidden investor behavior profiles. Users can either use a built-in
# sample investor dataset or upload their own tabular dataset. The app allows
# users to select numeric features, tune clustering parameters, and interpret
# results through K-Means clustering, PCA, hierarchical clustering, silhouette
# scores, elbow plots, dendrograms, and plain-English investor persona labels.
#
# Why this matters:
# In supervised learning, the model learns from an existing target label.
# In unsupervised learning, there is no target label. The goal is to discover
# patterns that may not be obvious at first. This app uses that idea to help
# users explore how investors may naturally group based on trading behavior,
# diversification, risk exposure, and portfolio allocation.

# 1. Imports

# os is used to build file paths that work locally and on Streamlit Cloud.
import os

# numpy and pandas are used for numerical operations and tabular data handling.
import numpy as np
import pandas as pd

# Streamlit is the framework used to build the interactive web app.
import streamlit as st

# Plotly is used for interactive visualizations such as histograms, heatmaps,
# PCA scatterplots, and elbow plots.
import plotly.express as px
import plotly.graph_objects as go

# Matplotlib is used for the hierarchical clustering dendrogram.
import matplotlib.pyplot as plt

# streamlit-option-menu is used to create the clean sidebar navigation menu.
from streamlit_option_menu import option_menu

# Scikit-learn tools used for preprocessing and unsupervised machine learning.
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# SciPy is used to calculate and display hierarchical clustering structure.
from scipy.cluster.hierarchy import linkage, dendrogram


# 2. Page Configuration

# Configure the browser tab title and make the app use the full page width.
st.set_page_config(
    page_title="Investor Persona Clustering App",
    layout="wide",
)

# 3. Custom Styling

# Custom CSS is used to make the app visually consistent with the user's
# previous Investment Risk Analyzer app. This creates cleaner spacing,
# styled metric cards, and a polished sidebar.
st.markdown(
    """
    <style>
    /* Limit the main content width and add vertical spacing */
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Light sidebar background to separate controls from main content */
    section[data-testid="stSidebar"] {
        background-color: #f5f6fa;
    }

    /* Card-style formatting for Streamlit metric boxes */
    div[data-testid="stMetric"] {
        border: 1px solid rgba(49, 51, 63, 0.10);
        border-radius: 14px;
        padding: 12px 16px;
        background-color: white;
    }

    /* Reusable card style for cluster interpretation text */
    .insight-card {
        border: 1px solid rgba(49, 51, 63, 0.12);
        border-radius: 14px;
        padding: 16px 18px;
        background-color: #ffffff;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# 4. File Paths and App Constants

# Get the folder where main.py is located. This makes file paths reliable
# whether the app is running locally or deployed on Streamlit Community Cloud.
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# The app checks for either a CSV or Excel version of the sample dataset.
# This gives flexibility depending on which file is included in GitHub.
SAMPLE_CSV_PATH = os.path.join(APP_DIR, "investor_persona_dataset_v2.csv")
SAMPLE_XLSX_PATH = os.path.join(APP_DIR, "investor_persona_dataset.xlsx")

# These are the intended numeric variables in the investor persona dataset.
# They describe investor behavior, portfolio allocation, and risk exposure.
# The app selects these by default because they are meaningful measures of
# investor similarity for clustering.
INVESTOR_FEATURES = [
    "portfolio_turnover",
    "avg_holding_period_days",
    "volatility_exposure",
    "diversification_score",
    "sector_concentration",
    "num_holdings",
    "cash_allocation_pct",
    "international_exposure_pct",
]

# These columns are useful for identification or explanation, but should not
# be selected automatically as machine learning features. For example,
# investor_id identifies a row but does not describe investor behavior, while
# cluster labels are outputs from the model rather than inputs.
NON_FEATURE_COLUMNS = [
    "investor_id",
    "true_persona",
    "persona",
    "cluster",
    "cluster_name",
    "kmeans_cluster",
    "hierarchical_cluster",
]

# 5. Data Creation, Loading, and Cleaning Functions

@st.cache_data
def create_sample_investor_dataset() -> pd.DataFrame:
    """
    Create a built-in sample investor dataset.

    This fallback dataset ensures the app always works, even if the external
    sample file is missing from the project folder. The dataset is intentionally
    designed with several investor archetypes plus a few noisy observations so
    that clustering results are interpretable but not unrealistically perfect.
    """
    np.random.seed(42)
    rows = []

    def add_cluster(
        n,
        persona,
        turnover,
        hold,
        vol,
        div,
        sector,
        holdings,
        cash,
        intl,
        noise=0.05,
    ):
        """
        Add one group of investors centered around a behavioral persona.

        Random variation is added around each persona profile so that the data
        feels more realistic and clustering is not perfectly separated.
        """
        for _ in range(n):
            rows.append(
                [
                    len(rows) + 1,
                    persona,
                    round(float(np.clip(np.random.normal(turnover, noise), 0, 1)), 4),
                    int(max(1, np.random.normal(hold, hold * 0.10))),
                    round(float(np.clip(np.random.normal(vol, noise), 0, 1)), 4),
                    round(float(np.clip(np.random.normal(div, noise), 0, 1)), 4),
                    round(float(np.clip(np.random.normal(sector, noise), 0, 1)), 4),
                    int(max(1, np.random.normal(holdings, 3))),
                    round(float(np.clip(np.random.normal(cash, noise), 0, 1)), 4),
                    round(float(np.clip(np.random.normal(intl, noise), 0, 1)), 4),
                ]
            )

    # Five investor personas are simulated so the app can reveal meaningful
    # segments through unsupervised learning.
    add_cluster(10, "Aggressive Trader", 0.85, 30, 0.90, 0.30, 0.80, 8, 0.05, 0.20)
    add_cluster(10, "Passive Long-Term Investor", 0.10, 380, 0.20, 0.85, 0.20, 30, 0.18, 0.45)
    add_cluster(10, "Balanced Investor", 0.50, 180, 0.50, 0.60, 0.50, 20, 0.10, 0.30)
    add_cluster(10, "Concentrated Growth Investor", 0.70, 90, 0.85, 0.40, 0.90, 10, 0.05, 0.20)
    add_cluster(10, "Conservative Wealth Preserver", 0.18, 320, 0.22, 0.75, 0.28, 27, 0.35, 0.38)

    # Add a few noisy/outlier investors to reflect real-world ambiguity.
    # These rows make the clustering task more realistic and less artificial.
    for _ in range(5):
        rows.append(
            [
                len(rows) + 1,
                "Noise / Outlier",
                round(float(np.random.uniform(0, 1)), 4),
                int(np.random.uniform(10, 500)),
                round(float(np.random.uniform(0, 1)), 4),
                round(float(np.random.uniform(0, 1)), 4),
                round(float(np.random.uniform(0, 1)), 4),
                int(np.random.uniform(5, 40)),
                round(float(np.random.uniform(0, 1)), 4),
                round(float(np.random.uniform(0, 1)), 4),
            ]
        )

    columns = [
        "investor_id",
        "true_persona",
        "portfolio_turnover",
        "avg_holding_period_days",
        "volatility_exposure",
        "diversification_score",
        "sector_concentration",
        "num_holdings",
        "cash_allocation_pct",
        "international_exposure_pct",
    ]

    return pd.DataFrame(rows, columns=columns)


@st.cache_data
def load_sample_data() -> pd.DataFrame:
    """
    Load the sample dataset used by the app.

    Loading priority:
    1. Use investor_persona_dataset_v2.csv if it exists.
    2. Use investor_persona_dataset.xlsx if it exists.
    3. Generate a built-in sample dataset if no file exists.

    This design helps prevent deployment errors on Streamlit Cloud.
    """
    # Prefer the CSV version because CSV files are lightweight and reliable
    # for Streamlit Community Cloud deployment.
    if os.path.exists(SAMPLE_CSV_PATH):
        return pd.read_csv(SAMPLE_CSV_PATH)

    # Fall back to Excel if the CSV version is not available.
    # This requires openpyxl in requirements.txt.
    if os.path.exists(SAMPLE_XLSX_PATH):
        return pd.read_excel(SAMPLE_XLSX_PATH)

    # Final fallback: generate a sample dataset directly from code so the app
    # still works even if no external sample data file is found.
    return create_sample_investor_dataset()


@st.cache_data
def get_sample_download_bytes() -> bytes:
    """
    Convert the sample dataset to CSV bytes for the download button.
    """
    sample_df = load_sample_data()
    return sample_df.to_csv(index=False).encode("utf-8")


@st.cache_data
def load_uploaded_data(uploaded_file):
    """
    Load a user-uploaded CSV or Excel file.

    Supporting both CSV and Excel makes the app easier for different users,
    since tabular datasets often come in either format.
    """
    if uploaded_file is None:
        return None

    file_name = uploaded_file.name.lower()

    # CSV files are read directly with pandas.
    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    # Excel files are also supported because many users store tabular data
    # in spreadsheet format.
    if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        return pd.read_excel(uploaded_file)

    return None


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset before analysis.

    This function standardizes column names and attempts to convert numeric-like
    columns into numeric data types. Text columns, such as true_persona, remain
    unchanged if they cannot be converted.
    """
    df = df.copy()

    # Strip spaces from column names so later feature selection is consistent.
    df.columns = [str(col).strip() for col in df.columns]

    # Convert columns to numeric when possible. This helps uploaded datasets
    # work properly with clustering and PCA.
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass

    return df


def safe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a dataframe for display in Streamlit.

    Some object columns can create display issues, so this function converts
    text columns to strings before showing them in st.dataframe().
    """
    display_df = df.copy()

    for col in display_df.columns:
        if display_df[col].dtype == "object":
            display_df[col] = display_df[col].fillna("").astype(str)

    return display_df


def get_numeric_columns(df: pd.DataFrame) -> list:
    """
    Return numeric columns that contain at least one non-null value.

    Clustering, PCA, and correlation analysis require numeric inputs.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if df[col].notna().sum() > 0]


def get_default_features(df: pd.DataFrame) -> list:
    """
    Choose default machine learning features.

    If the sample investor behavior features are available, the app selects
    them by default. If a user uploads a different dataset, the app falls back
    to other numeric columns while avoiding obvious ID or label columns.
    """
    numeric_cols = get_numeric_columns(df)

    # Prioritize the intended investor behavior variables.
    investor_defaults = [col for col in INVESTOR_FEATURES if col in numeric_cols]
    if len(investor_defaults) >= 2:
        return investor_defaults

    # Fallback for uploaded datasets with different column names.
    excluded = [col.lower() for col in NON_FEATURE_COLUMNS]
    fallback = [col for col in numeric_cols if col.lower() not in excluded]

    return fallback[: min(8, len(fallback))]


# 6. Machine Learning Helper Functions

def prepare_model_data(df: pd.DataFrame, selected_features: list, scale_data: bool = True):
    """
    Prepare selected columns for unsupervised machine learning.

    Processing steps:
    1. Select the user-chosen numeric features.
    2. Fill missing values using the median.
    3. Optionally standardize values using StandardScaler.

    Scaling is important because K-Means and hierarchical clustering are
    distance-based. Without scaling, a feature measured in large units
    such as holding period days could overpower percentage-based features.
    """
    # Create a copy of selected columns so the original dataframe remains
    # unchanged for display, downloads, and other app pages.
    X_raw = df[selected_features].copy()

    # Impute missing values so clustering algorithms do not fail on incomplete
    # data. Median imputation is used because it is simple and less sensitive
    # to extreme values than mean imputation.
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_raw)

    # Standardize features if selected by the user.
    # This matters because K-Means and hierarchical clustering are distance-based.
    # Without scaling, large-unit columns such as holding period days could
    # overpower percentage-based columns such as portfolio turnover.
    if scale_data:
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X_imputed)
    else:
        X_processed = X_imputed

    # Return both versions:
    # - X_raw supports interpretation and display.
    # - X_processed is used by the machine learning models.
    return X_raw, X_processed


def run_kmeans(X_processed, k: int, init_method: str = "k-means++"):
    """
    Fit a K-Means clustering model.

    K-Means groups observations by minimizing the distance between each
    observation and its assigned cluster center.

    Parameters:
        X_processed: Prepared numeric feature matrix.
        k (int): Number of clusters selected by the user.
        init_method (str): Centroid initialization strategy. Users can compare
                           "k-means++" and "random" to see how initialization
                           can affect clustering results.
    """
    # random_state makes results reproducible across app reruns.
    # n_init=10 runs K-Means several times and keeps the best result,
    # which improves stability.
    model = KMeans(
        n_clusters=k,
        init=init_method,
        random_state=42,
        n_init=10,
    )

    # fit_predict both trains the model and assigns each investor to a cluster.
    labels = model.fit_predict(X_processed)

    return labels, model


def calculate_silhouette(X_processed, labels):
    """
    Calculate silhouette score for clustering quality.

    Silhouette score compares how close an observation is to its own cluster
    versus other clusters. Scores closer to 1 suggest clearer separation,
    while scores near 0 suggest overlapping clusters.
    """
    unique_labels = np.unique(labels)

    # Silhouette score requires at least 2 clusters and fewer clusters than rows.
    if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        return np.nan

    return silhouette_score(X_processed, labels)


def display_silhouette_interpretation(score):
    """
    Display a plain-English interpretation of the silhouette score.

    This helps users understand whether the clustering output appears strong,
    moderate, or weak instead of only seeing a numeric metric.
    """
    if pd.isna(score):
        st.info("Silhouette score is not available for the current clustering setup.")
    elif score >= 0.50:
        st.success("Strong clustering structure detected. The clusters appear well separated.")
    elif score >= 0.25:
        st.info("Moderate clustering structure detected. Some clusters may overlap, but meaningful grouping is still present.")
    else:
        st.warning("Weak clustering structure detected. The selected features or number of clusters may not separate the data clearly.")


def display_silhouette_interpretation(score):
    """
    Display a plain-English interpretation of the silhouette score.

    This helps users understand whether the clustering output appears strong,
    moderate, or weak instead of only seeing a numeric metric.
    """
    if pd.isna(score):
        st.info("Silhouette score is not available for the current clustering setup.")
    elif score >= 0.50:
        st.success("Strong clustering structure detected. The clusters appear well separated.")
    elif score >= 0.25:
        st.info("Moderate clustering structure detected. Some clusters may overlap, but meaningful grouping is still present.")
    else:
        st.warning("Weak clustering structure detected. The selected features or number of clusters may not separate the data clearly.")


def create_elbow_plot(X_processed, max_k: int = 10):
    """
    Create an elbow plot for K-Means.

    The elbow plot shows inertia across different k values. Inertia measures
    within-cluster compactness. A useful k often appears where the curve begins
    flattening, meaning additional clusters provide smaller improvements.
    """
    max_k = min(max_k, len(X_processed) - 1)

    if max_k < 2:
        return None

    k_values = list(range(2, max_k + 1))
    inertia_values = []

    # Fit K-Means repeatedly with different k values.
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X_processed)
        inertia_values.append(model.inertia_)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=k_values,
            y=inertia_values,
            mode="lines+markers",
            name="Inertia",
        )
    )

    fig.update_layout(
        title="Elbow Plot",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Inertia",
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


def create_pca_dataframe(X_processed, labels=None):
    """
    Reduce the dataset to two principal components.

    PCA transforms many numeric features into fewer components that preserve
    as much variance as possible. This makes it easier to visualize clusters
    in a two-dimensional scatterplot.
    """
    # Use two components because the goal is to create a 2D visualization
    # that users can easily interpret.
    pca = PCA(n_components=2)

    # fit_transform learns the principal components and transforms the original
    # high-dimensional feature matrix into PC1 and PC2 coordinates.
    components = pca.fit_transform(X_processed)

    # Store PCA coordinates in a dataframe for easy plotting with Plotly.
    pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])

    # Add cluster labels if provided so the PCA plot can be color-coded.
    if labels is not None:
        pca_df["Cluster"] = labels.astype(str)

    return pca_df, pca


def create_pca_scatter(pca_df: pd.DataFrame, title: str):
    """
    Create an interactive PCA scatterplot.
    """
    if "Cluster" in pca_df.columns:
        fig = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            color="Cluster",
            title=title,
            template="plotly_white",
        )
    else:
        fig = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            title=title,
            template="plotly_white",
        )

    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


def create_cluster_summary(df: pd.DataFrame, selected_features: list, labels, cluster_col: str):
    """
    Summarize average feature values by cluster.

    This table helps translate algorithmic cluster labels into meaningful
    behavioral patterns by showing what each group looks like on average.
    """
    # Combine selected features with cluster labels so we can summarize each
    # group created by the model.
    summary_df = df[selected_features].copy()
    summary_df[cluster_col] = labels

    # Mean feature values reveal what makes each cluster different.
    # Rounding keeps the table readable in the app.
    summary = summary_df.groupby(cluster_col)[selected_features].mean().round(3)

    # Count shows how many observations belong to each cluster.
    summary["count"] = summary_df.groupby(cluster_col).size()

    # Put count first so users can immediately see cluster size.
    summary = summary[["count"] + selected_features]

    return summary.reset_index()


def describe_cluster(row: pd.Series) -> str:
    """
    Generate a plain-English interpretation of a cluster.

    These rules are designed around the investor persona dataset. The goal is
    to make the model output easier to understand for non-technical users.
    """
    turnover = row.get("portfolio_turnover", np.nan)
    holding = row.get("avg_holding_period_days", np.nan)
    volatility = row.get("volatility_exposure", np.nan)
    diversification = row.get("diversification_score", np.nan)
    sector = row.get("sector_concentration", np.nan)
    cash = row.get("cash_allocation_pct", np.nan)

    if pd.notna(turnover) and pd.notna(volatility) and pd.notna(holding):
        if turnover >= 0.65 and volatility >= 0.70 and holding <= 120:
            return "Aggressive Trader: high trading activity, short holding periods, and elevated volatility exposure."

    if pd.notna(turnover) and pd.notna(diversification) and pd.notna(holding):
        if turnover <= 0.25 and diversification >= 0.70 and holding >= 250:
            return "Passive Long-Term Investor: low turnover, longer holding periods, and strong diversification."

    if pd.notna(sector) and pd.notna(volatility):
        if sector >= 0.75 and volatility >= 0.70:
            return "Concentrated Growth Investor: high sector concentration with elevated risk exposure."

    if pd.notna(cash) and pd.notna(volatility):
        if cash >= 0.25 and volatility <= 0.35:
            return "Conservative Wealth Preserver: higher cash allocation and lower volatility exposure."

    return "Balanced or Mixed Investor: moderate behavior across several portfolio characteristics."


def name_cluster(row: pd.Series) -> str:
    """
    Assign a short, user-friendly investor persona name to each cluster.

    This function uses the average feature values from each cluster summary
    table to translate numeric model output into language that is easier for
    non-technical users to understand.
    """
    turnover = row.get("portfolio_turnover", np.nan)
    holding = row.get("avg_holding_period_days", np.nan)
    volatility = row.get("volatility_exposure", np.nan)
    diversification = row.get("diversification_score", np.nan)
    sector = row.get("sector_concentration", np.nan)
    cash = row.get("cash_allocation_pct", np.nan)

    if pd.notna(turnover) and pd.notna(volatility) and pd.notna(holding):
        if turnover >= 0.65 and volatility >= 0.70 and holding <= 120:
            return "Aggressive Traders"

    if pd.notna(turnover) and pd.notna(diversification) and pd.notna(holding):
        if turnover <= 0.25 and diversification >= 0.70 and holding >= 250:
            return "Passive Long-Term Investors"

    if pd.notna(sector) and pd.notna(volatility):
        if sector >= 0.75 and volatility >= 0.70:
            return "Concentrated Growth Investors"

    if pd.notna(cash) and pd.notna(volatility):
        if cash >= 0.25 and volatility <= 0.35:
            return "Conservative Wealth Preservers"

    return "Balanced / Mixed Investors"


def create_correlation_heatmap(df: pd.DataFrame, numeric_cols: list):
    """
    Create a correlation matrix heatmap.

    The heatmap helps users see which numeric variables move together before
    running clustering or PCA.
    """
    corr = df[numeric_cols].corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation Matrix",
    )

    fig.update_layout(
        height=650,
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


def create_dendrogram(X_processed, labels=None):
    """
    Create a hierarchical clustering dendrogram.

    The dendrogram shows how observations merge together based on similarity.
    Ward linkage is used because it tends to create compact, interpretable
    clusters for numeric data.
    """
    linked = linkage(X_processed, method="ward")

    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(
        linked,
        ax=ax,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=8,
    )

    ax.set_title("Hierarchical Clustering Dendrogram")
    ax.set_xlabel("Observations")
    ax.set_ylabel("Distance")

    plt.tight_layout()

    return fig


# 7. Sidebar Navigation and Dataset Controls

with st.sidebar:
    st.markdown("## Menu")

    # Sidebar navigation mirrors the layout style of the previous app while
    # changing the pages to match the unsupervised learning assignment.
    page = option_menu(
        "Navigation",
        ["Overview", "Explore Data", "Clustering", "PCA Analysis", "Hierarchical Analysis"],
        icons=["house", "bar-chart", "diagram-3", "graph-up", "share"],
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
                "font-size": "18px",
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

    # Let users choose between the built-in investor dataset and their own file.
    # The data source selector lets users either use the curated sample dataset
    # or upload a custom dataset for their own clustering analysis.
    data_source = st.radio(
        "Choose a data source:",
        ["Use sample investor dataset", "Upload my own dataset"],
    )

    uploaded_file = None

    # File uploader appears only when the user chooses to upload data.
    if data_source == "Upload my own dataset":
        uploaded_file = st.file_uploader(
            "Upload a CSV or Excel file",
            type=["csv", "xlsx", "xls"],
        )

    # Give users access to the sample dataset so they can inspect or reuse it.
    st.download_button(
        label="Download sample dataset",
        data=get_sample_download_bytes(),
        file_name="investor_persona_dataset.csv",
        mime="text/csv",
    )

    st.caption(
        "The sample dataset includes simulated investor behavior profiles with realistic variation and a few noisy observations."
    )


# ------------------------------------------------------------
# 8. Load the Selected Dataset
# ------------------------------------------------------------

df = None

# Load the sample dataset when selected.
if data_source == "Use sample investor dataset":
    df = load_sample_data()
    df = clean_dataframe(df)

# Load the uploaded dataset when the user provides a file.
elif data_source == "Upload my own dataset":
    if uploaded_file is not None:
        df = load_uploaded_data(uploaded_file)
        if df is not None:
            df = clean_dataframe(df)


# ------------------------------------------------------------
# 9. Main App Header
# ------------------------------------------------------------

st.title("Investor Persona Clustering App")

st.write(
    "This app uses unsupervised machine learning to uncover hidden investor profiles based on portfolio behavior, trading patterns, diversification, and risk exposure."
)


# ------------------------------------------------------------
# 10. Overview Page
# ------------------------------------------------------------

if page == "Overview":
    st.header("Overview")

    st.write(
        """
        This project demonstrates unsupervised machine learning by grouping investors
        into behavior-based segments. Instead of predicting a known label, the app
        discovers hidden structure in investor data using K-Means clustering,
        Principal Component Analysis, and hierarchical clustering.
        """
    )

    with st.expander("Why use unsupervised learning for investor behavior?"):
        st.write(
            """
            Unsupervised learning is useful when there is no predefined target label.
            Instead of telling the model what type of investor each row represents,
            the app asks the model to discover natural groupings based on behavior.

            This approach is helpful for investor segmentation because investors may
            share similar patterns in turnover, holding period, diversification, and
            risk exposure even if those groups are not labeled in advance.
            """
        )

    if df is None:
        st.warning("Please select the sample dataset or upload your own dataset.")
    else:
        numeric_cols = get_numeric_columns(df)
        default_features = get_default_features(df)

        # High-level dataset metrics give users quick context.
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric("Rows", f"{df.shape[0]}")
        with m2:
            st.metric("Columns", f"{df.shape[1]}")
        with m3:
            st.metric("Numeric Columns", f"{len(numeric_cols)}")
        with m4:
            st.metric("Default ML Features", f"{len(default_features)}")

        st.markdown("---")

        left, right = st.columns(2)

        with left:
            st.subheader("Dataset Preview")
            st.dataframe(safe_for_display(df.head(10)), use_container_width=True)

        with right:
            st.subheader("App Capabilities")
            st.markdown(
                """
                - Upload a tabular dataset or use the built-in investor sample dataset.
                - Select numeric features for unsupervised learning.
                - Adjust the number of clusters used in K-Means and hierarchical clustering.
                - Evaluate clustering with silhouette scores and elbow plots.
                - Visualize patterns using PCA scatterplots and dendrograms.
                - Download the clustered dataset for further analysis.
                """
            )

        st.markdown("---")

        st.subheader("Sample Dataset Features")

        # This table helps explain the meaning of the sample dataset fields.
        feature_info = pd.DataFrame(
            {
                "Feature": INVESTOR_FEATURES,
                "Description": [
                    "Frequency of trading activity.",
                    "Average number of days assets are held.",
                    "Estimated exposure to portfolio volatility.",
                    "Degree of diversification across holdings.",
                    "Extent of concentration in a single sector.",
                    "Number of individual assets in the portfolio.",
                    "Percentage of portfolio held in cash.",
                    "Percentage of portfolio invested internationally.",
                ],
            }
        )

        st.dataframe(feature_info, use_container_width=True)


# ------------------------------------------------------------
# 11. Explore Data Page
# ------------------------------------------------------------

elif page == "Explore Data":
    st.header("Explore Data")

    if df is None:
        st.warning("Please select the sample dataset or upload your own dataset.")
    else:
        numeric_cols = get_numeric_columns(df)

        left, right = st.columns(2)

        with left:
            st.subheader("Dataset Preview")
            st.dataframe(safe_for_display(df.head(25)), use_container_width=True)

        with right:
            st.subheader("Summary Statistics")
            if numeric_cols:
                summary_df = df[numeric_cols].describe().transpose().round(3)
                st.dataframe(safe_for_display(summary_df), use_container_width=True)
            else:
                st.info("No numeric columns are available for summary statistics.")

        st.markdown("---")

        # Histogram allows users to understand the shape and spread of one feature.
        if numeric_cols:
            st.subheader("Feature Distribution")

            graph_col = st.selectbox(
                "Choose a numeric column to graph",
                numeric_cols,
            )

            fig = px.histogram(
                df,
                x=graph_col,
                nbins=20,
                title=f"Distribution of {graph_col}",
                template="plotly_white",
            )

            fig.update_layout(
                height=420,
                margin=dict(l=20, r=20, t=50, b=20),
            )

            st.plotly_chart(fig, use_container_width=True)

            with st.expander("How should I interpret this feature distribution?"):
                st.write(
                    """
                    The histogram shows how values for the selected feature are spread
                    across investors. Peaks may suggest common behavior patterns, while
                    gaps or separate groups may suggest that investors naturally fall
                    into different segments.

                    This view is useful before clustering because it helps users understand
                    whether a feature has enough variation to help separate investor profiles.
                    """
                )

        st.markdown("---")

        # Correlation analysis helps users detect relationships between features.
        if len(numeric_cols) >= 2:
            st.subheader("Correlation Matrix")

            fig = create_correlation_heatmap(df, numeric_cols)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Why look at correlations before clustering?"):
                st.write(
                    """
                    The correlation matrix shows relationships between numeric features.
                    Strong positive or negative correlations can reveal features that move
                    together or capture similar information.

                    This matters because clustering is affected by the features selected.
                    If several features are highly related, they may give extra weight to
                    the same underlying behavior. Reviewing correlations helps users make
                    more thoughtful feature choices.
                    """
                )
        else:
            st.info("At least two numeric columns are needed for a correlation matrix.")


# 12. K-Means Clustering Page

elif page == "Clustering":
    st.header("K-Means Clustering")

    if df is None:
        st.warning("Please select the sample dataset or upload your own dataset.")
    else:
        numeric_cols = get_numeric_columns(df)
        default_features = get_default_features(df)

        if len(numeric_cols) < 2:
            st.error("You need at least two numeric columns for clustering.")
        else:
            st.write(
                "Use this section to experiment with K-Means clustering. Select features, adjust the number of clusters, compare initialization methods, and observe how the results change."
            )

            st.info(
                "K-Means is a distance-based clustering method. Changing the selected features, standardization setting, initialization method, or k value can change the final investor groupings."
            )

            with st.expander("Why use K-Means for this app?"):
                st.write(
                    """
                    K-Means is useful for quickly grouping investors into a chosen number
                    of segments. It works well when the goal is to create clear, interpretable
                    groups based on numeric behavior patterns.

                    In this app, K-Means helps translate portfolio behavior into potential
                    investor personas, such as active traders, passive long-term investors,
                    or conservative wealth preservers.
                    """
                )

            with st.expander("Why use K-Means in this application?"):
                st.write(
                    """
                    K-Means is used because it creates clear, interpretable groups based on
                    similarity. It is especially useful when the goal is to segment investors
                    into a fixed number of categories, such as investor personas.

                    Compared with more complex clustering methods, K-Means is efficient,
                    easy to explain, and works well when clusters are relatively separated.
                    This makes it a strong choice for an interactive app where users need to
                    quickly understand how investor groups are formed.
                    """
                )

            # Users choose which numeric variables should define similarity.
            selected_features = st.multiselect(
                "Select features for clustering",
                options=numeric_cols,
                default=default_features,
            )

            if len(selected_features) < 2:
                st.warning("Please select at least two numeric features.")
            else:
                c1, c2, c3 = st.columns(3)

                with c1:
                    # k is the main hyperparameter for K-Means.
                    k = st.slider(
                        "Number of clusters (k)",
                        min_value=2,
                        max_value=min(10, len(df) - 1),
                        value=min(5, min(10, len(df) - 1)),
                        help="Controls how many investor groups the model should create.",
                    )

                with c2:
                    # Initialization affects where K-Means begins placing centroids.
                    # Initialization controls where K-Means begins placing
                    # cluster centers. Allowing the user to change this supports
                    # experimentation with model sensitivity.
                    init_method = st.selectbox(
                        "Initialization Method",
                        ["k-means++", "random"],
                        help="k-means++ usually creates more stable starting points, while random initialization allows users to test sensitivity.",
                    )

                with c3:
                    # Scaling is recommended because clustering depends on distances.
                    scale_data = st.checkbox(
                        "Standardize features",
                        value=True,
                        help="Recommended because clustering is distance-based.",
                    )

                # Prepare the feature matrix and run K-Means.
                X_raw, X_processed = prepare_model_data(df, selected_features, scale_data)
                labels, kmeans_model = run_kmeans(X_processed, k, init_method)
                silhouette = calculate_silhouette(X_processed, labels)

                # Add cluster assignments back to the original dataframe.
                clustered_df = df.copy()
                clustered_df["kmeans_cluster"] = labels

                st.subheader("Model Feedback")

                # Key metrics summarize the clustering setup and output.
                m1, m2, m3, m4, m5 = st.columns(5)

                with m1:
                    st.metric("Selected Features", f"{len(selected_features)}")
                with m2:
                    st.metric("Clusters", f"{k}")
                with m3:
                    st.metric("Init Method", init_method)
                with m4:
                    st.metric(
                        "Silhouette Score",
                        f"{silhouette:.3f}" if pd.notna(silhouette) else "N/A",
                    )
                with m5:
                    st.metric("Rows Clustered", f"{len(clustered_df)}")

                st.caption(
                    "Silhouette score ranges from -1 to 1. Higher values usually indicate more distinct clusters."
                )

                # Provide interpretation so users understand the metric in context.
                display_silhouette_interpretation(silhouette)

                with st.expander("What does the silhouette score mean?"):
                    st.write(
                        """
                        The silhouette score measures how well each investor fits within
                        its assigned cluster compared with other clusters. Scores closer
                        to 1 suggest that clusters are clearly separated. Scores near 0
                        suggest overlap between groups, while negative scores may indicate
                        that some observations may fit better in another cluster.

                        This metric is useful for unsupervised learning because there is
                        no true target label to compare predictions against.
                        """
                    )

                with st.expander("Why choose these K-Means settings?"):
                    st.write(
                        """
                        The number of clusters, initialization method, feature selection,
                        and scaling option all affect the clustering results.

                        - Number of clusters (k): controls how many investor groups are created.
                        - Initialization method: controls how starting cluster centers are chosen.
                        - Feature selection: determines which investor behaviors define similarity.
                        - Standardization: makes features comparable when they use different units.

                        These controls are included so users can experiment and see how modeling
                        choices change the final investor segments.
                        """
                    )

                # This expander directly addresses how hyperparameters affect
                # unsupervised model behavior and interpretation.
                with st.expander("Why do hyperparameters matter in clustering?"):
                    st.write(
                        """
                        Hyperparameters control how the clustering algorithm behaves and can
                        directly change the final investor groups.

                        - **Number of clusters (k):** determines how many investor segments the model creates.
                        - **Feature selection:** defines what similarity means for the model.
                        - **Standardization:** prevents large-scale variables, such as holding period days, from overpowering percentage-based features.
                        - **Initialization method:** affects where K-Means begins placing cluster centers.

                        Because unsupervised learning does not have one correct answer, experimenting
                        with these settings helps users decide which clustering solution is most useful
                        and interpretable.
                        """
                    )

                st.info(
                    "Clustering results should be interpreted in context. There is no single correct grouping; there are useful patterns based on the selected features and settings."
                )

                # Tabs organize the major clustering outputs clearly.
                tab1, tab2, tab3, tab4 = st.tabs(
                    ["Cluster Summary", "PCA Cluster Plot", "Elbow Plot", "Clustered Dataset"]
                )

                with tab1:
                    st.subheader("Cluster Summary")

                    summary = create_cluster_summary(
                        clustered_df,
                        selected_features,
                        labels,
                        cluster_col="kmeans_cluster",
                    )

                    st.dataframe(summary, use_container_width=True)

                    with st.expander("How should I read the cluster summary table?"):
                        st.write(
                            """
                            The cluster summary table shows the average value of each selected
                            feature within each cluster. This makes it easier to understand
                            what separates one investor group from another.

                            For example, a cluster with high portfolio turnover and short
                            holding periods may represent active traders, while a cluster with
                            low turnover and high diversification may represent passive long-term
                            investors.
                            """
                        )

                    st.subheader("Investor Personas Identified by the Model")

                    st.write(
                        "Based on the selected features and clustering settings, the model identified the following investor profiles:"
                    )

                    # Translate numeric clusters into plain-English investor profiles.
                    for _, row in summary.iterrows():
                        cluster_id = int(row["kmeans_cluster"])
                        cluster_name = name_cluster(row)
                        interpretation = describe_cluster(row)

                        st.markdown(
                            f"""
                            <div class="insight-card">
                                <strong>Cluster {cluster_id}: {cluster_name}</strong><br>
                                {interpretation}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with st.expander("How should I use these persona labels?"):
                        st.write(
                            """
                            The persona labels are interpretations based on the average feature
                            values within each cluster. They are not official investor categories
                            or predictions. Instead, they help translate the clustering output into
                            more understandable business language.

                            Users should compare the persona label with the cluster summary table
                            to confirm whether the interpretation makes sense for the selected
                            features and clustering settings.
                            """
                        )

                with tab2:
                    st.subheader("PCA Visualization of K-Means Clusters")

                    # PCA lets users visualize multi-feature clusters in 2D.
                    pca_df, pca_model = create_pca_dataframe(X_processed, labels)
                    fig = create_pca_scatter(
                        pca_df,
                        "K-Means Clusters Projected with PCA",
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    variance = pca_model.explained_variance_ratio_

                    v1, v2, v3 = st.columns(3)
                    with v1:
                        st.metric("PC1 Explained Variance", f"{variance[0]:.2%}")
                    with v2:
                        st.metric("PC2 Explained Variance", f"{variance[1]:.2%}")
                    with v3:
                        st.metric("Total Explained", f"{variance.sum():.2%}")

                    with st.expander("What do PC1 and PC2 explained variance mean?"):
                        st.write(
                            """
                            Principal Component Analysis (PCA) transforms the original features
                            into new variables called principal components.

                            - **PC1 (Principal Component 1)** captures the largest amount of variation in the data.
                            - **PC2 (Principal Component 2)** captures the second largest amount of variation.
                            - **Total Explained Variance** shows how much information is preserved by PC1 and PC2 together.

                            A higher total explained variance means the 2D visualization is a stronger
                            summary of the original selected features.
                            """
                        )

                with tab3:
                    st.subheader("Elbow Plot")

                    elbow_fig = create_elbow_plot(X_processed, max_k=10)

                    if elbow_fig is not None:
                        st.plotly_chart(elbow_fig, use_container_width=True)
                        st.write(
                            "The elbow plot helps compare different k values. A useful k often appears where inertia begins decreasing more slowly."
                        )
                        st.caption(
                            "If the curve has a clear bend, that point may be a reasonable balance between compact clusters and interpretability."
                        )

                        with st.expander("How should I choose the number of clusters?"):
                            st.write(
                                """
                                The elbow plot shows how inertia changes as the number
                                of clusters increases. Inertia measures how tightly points
                                fit within their assigned clusters.

                                A good k is often near the point where the curve begins
                                to flatten. That means adding more clusters only slightly
                                improves compactness, so the simpler solution may be easier
                                to interpret.
                                """
                            )
                    else:
                        st.info("Not enough observations to create an elbow plot.")

                with tab4:
                    st.subheader("Dataset with K-Means Cluster Assignments")

                    st.dataframe(safe_for_display(clustered_df), use_container_width=True)

                    # Export the dataset with K-Means labels so users can
                    # continue analyzing the results outside the app.
                    st.download_button(
                        label="Download clustered dataset",
                        data=clustered_df.to_csv(index=False).encode("utf-8"),
                        file_name="investor_persona_kmeans_clusters.csv",
                        mime="text/csv",
                    )


# 13. PCA Analysis Page

elif page == "PCA Analysis":
    st.header("Principal Component Analysis")

    if df is None:
        st.warning("Please select the sample dataset or upload your own dataset.")
    else:
        numeric_cols = get_numeric_columns(df)
        default_features = get_default_features(df)

        if len(numeric_cols) < 2:
            st.error("You need at least two numeric columns for PCA.")
        else:
            st.write(
                "PCA reduces multiple numeric features into two main components, making it easier to visualize hidden structure in the dataset."
            )

            # This expander explains why PCA was selected and connects the
            # technical method to the user-facing visualization.
            with st.expander("Why choose PCA for this app?"):
                st.write(
                    """
                    PCA is used because investor behavior is described by multiple features,
                    which are difficult to visualize all at once. PCA reduces those features
                    into two main components while preserving as much variation as possible.

                    This makes PCA useful for communicating clustering results because users
                    can see whether investors with similar behavior appear close together in
                    a simplified two-dimensional space.
                    """
                )

            selected_features = st.multiselect(
                "Select features for PCA",
                options=numeric_cols,
                default=default_features,
            )

            if len(selected_features) < 2:
                st.warning("Please select at least two numeric features.")
            else:
                scale_data = st.checkbox(
                    "Standardize features before PCA",
                    value=True,
                    help="Recommended when features use different units.",
                )

                # Prepare data before PCA. Standardization is especially important
                # because PCA is influenced by the scale of each variable.
                X_raw, X_processed = prepare_model_data(df, selected_features, scale_data)
                pca_df, pca_model = create_pca_dataframe(X_processed)

                st.subheader("PCA Scatter Plot")

                fig = create_pca_scatter(
                    pca_df,
                    "Two-Dimensional PCA Projection",
                )

                st.plotly_chart(fig, use_container_width=True)

                variance = pca_model.explained_variance_ratio_

                st.subheader("Explained Variance")

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("PC1", f"{variance[0]:.2%}")
                with m2:
                    st.metric("PC2", f"{variance[1]:.2%}")
                with m3:
                    st.metric("Total Explained", f"{variance.sum():.2%}")

                with st.expander("How should I interpret these PCA metrics?"):
                    st.write(
                        """
                        - **PC1 Explained Variance**: The percentage of total variation captured
                          by the first principal component. A higher value means PC1 captures more
                          of the dataset's structure.

                        - **PC2 Explained Variance**: The percentage of variation captured by the
                          second principal component. This adds additional detail beyond PC1.

                        - **Total Explained Variance**: The combined percentage of information
                          captured by PC1 and PC2. Higher values indicate that the 2D plot is a
                          stronger representation of the original selected features.
                        """
                    )

                with st.expander("Why does explained variance matter?"):
                    st.write(
                        """
                        Explained variance tells users how much information from the original
                        selected features is captured by the PCA plot. If PC1 and PC2 explain a
                        high percentage of the variance, then the 2D plot is a better summary of
                        the original feature space.

                        If the total explained variance is low, the PCA plot can still be useful,
                        but users should be more cautious because some information is not visible
                        in the two-dimensional view.
                        """
                    )

                st.subheader("PCA Feature Loadings")

                # Loadings show how much each original feature contributes to
                # PC1 and PC2, making PCA easier to interpret.
                loadings = pd.DataFrame(
                    pca_model.components_.T,
                    columns=["PC1 Loading", "PC2 Loading"],
                    index=selected_features,
                ).round(3)

                st.dataframe(loadings, use_container_width=True)

                st.caption(
                    "Feature loadings show how strongly each original variable contributes to each principal component."
                )

                st.info(
                    "A higher total explained variance means the two-dimensional PCA plot preserves more information from the original selected features."
                )

                with st.expander("What do PCA feature loadings mean?"):
                    st.write(
                        """
                        Feature loadings show how strongly each original feature contributes
                        to PC1 and PC2. Larger absolute values mean the feature has a stronger
                        influence on that component.

                        This helps explain which investor behaviors are driving the PCA
                        visualization. For example, if volatility exposure has a large loading
                        on PC1, then PC1 may be strongly related to risk exposure.
                        """
                    )


# 14. Hierarchical Clustering Page

elif page == "Hierarchical Analysis":
    st.header("Hierarchical Clustering")

    if df is None:
        st.warning("Please select the sample dataset or upload your own dataset.")
    else:
        numeric_cols = get_numeric_columns(df)
        default_features = get_default_features(df)

        if len(numeric_cols) < 2:
            st.error("You need at least two numeric columns for hierarchical clustering.")
        else:
            st.write(
                "Hierarchical clustering shows how observations relate to each other through a tree-like structure called a dendrogram."
            )

            with st.expander("Why use hierarchical clustering in addition to K-Means?"):
                st.write(
                    """
                    K-Means requires the user to choose a number of clusters before the
                    model runs. Hierarchical clustering shows how observations group together
                    step by step, which can help users understand the structure of the data
                    before deciding how many clusters make sense.

                    Using both methods provides a richer view of investor behavior because
                    K-Means gives clear segment assignments, while the dendrogram reveals
                    relationships between observations.
                    """
                )

            # This expander explains why hierarchical clustering is included
            # alongside K-Means.
            with st.expander("Why choose hierarchical clustering?"):
                st.write(
                    """
                    Hierarchical clustering is included because it shows the relationships
                    between investors before reducing them into final groups. This makes it
                    useful for exploring whether certain investors are very similar, loosely
                    related, or clearly separated.

                    It complements K-Means because K-Means gives a fast final segmentation,
                    while hierarchical clustering provides a more detailed view of how those
                    groups may naturally form.
                    """
                )

            selected_features = st.multiselect(
                "Select features for hierarchical clustering",
                options=numeric_cols,
                default=default_features,
            )

            if len(selected_features) < 2:
                st.warning("Please select at least two numeric features.")
            else:
                c1, c2 = st.columns(2)

                with c1:
                    # Number of clusters controls where the hierarchy is cut.
                    num_clusters = st.slider(
                        "Number of clusters",
                        min_value=2,
                        max_value=min(10, len(df) - 1),
                        value=min(5, min(10, len(df) - 1)),
                    )

                with c2:
                    scale_data = st.checkbox(
                        "Standardize features",
                        value=True,
                        help="Recommended because hierarchical clustering is distance-based.",
                    )

                X_raw, X_processed = prepare_model_data(df, selected_features, scale_data)

                # Agglomerative clustering builds clusters from the bottom up,
                # repeatedly merging similar observations or groups.
                # Agglomerative clustering builds groups from the bottom up by
                # repeatedly merging similar observations or clusters.
                # Ward linkage attempts to minimize within-cluster variance,
                # which often produces compact, interpretable groups.
                hierarchical_model = AgglomerativeClustering(
                    n_clusters=num_clusters,
                    linkage="ward",
                )

                # Assign observations to hierarchical clusters and evaluate
                # separation using the same silhouette metric as K-Means.
                labels = hierarchical_model.fit_predict(X_processed)
                silhouette = calculate_silhouette(X_processed, labels)

                clustered_df = df.copy()
                clustered_df["hierarchical_cluster"] = labels

                m1, m2, m3 = st.columns(3)

                with m1:
                    st.metric("Clusters", f"{num_clusters}")
                with m2:
                    st.metric("Selected Features", f"{len(selected_features)}")
                with m3:
                    st.metric(
                        "Silhouette Score",
                        f"{silhouette:.3f}" if pd.notna(silhouette) else "N/A",
                    )

                # Provide interpretation for hierarchical clustering quality.
                display_silhouette_interpretation(silhouette)

                with st.expander("What does the hierarchical silhouette score mean?"):
                    st.write(
                        """
                        The silhouette score is interpreted the same way for hierarchical
                        clustering as it is for K-Means. It measures whether investors are
                        closer to their assigned cluster than to other clusters.

                        This gives a metric-based way to compare whether the hierarchical
                        clustering solution appears strong, moderate, or weak.
                        """
                    )

                st.subheader("Dendrogram")

                # If investor IDs are available, use them as dendrogram labels
                # to make the tree easier to interpret.
                dendrogram_labels = None
                if "investor_id" in df.columns:
                    dendrogram_labels = df["investor_id"].astype(str).tolist()

                fig = create_dendrogram(X_processed, labels=dendrogram_labels)
                st.pyplot(fig)

                with st.expander("How should I interpret the dendrogram?"):
                    st.write(
                        """
                        The dendrogram shows how investors are merged into groups based
                        on similarity. Investors that merge lower on the chart are more
                        similar to each other. Merges that happen higher on the chart
                        represent larger differences between groups.

                        This visualization is useful because it shows the nested structure
                        of the data rather than only showing final cluster labels.
                        """
                    )

                st.subheader("Hierarchical Cluster Summary")

                summary = create_cluster_summary(
                    clustered_df,
                    selected_features,
                    labels,
                    cluster_col="hierarchical_cluster",
                )

                st.dataframe(summary, use_container_width=True)

                st.subheader("Hierarchical Investor Personas Identified by the Model")

                st.write(
                    "These labels provide a plain-English interpretation of each hierarchical cluster based on average feature values:"
                )

                for _, row in summary.iterrows():
                    cluster_id = int(row["hierarchical_cluster"])
                    cluster_name = name_cluster(row)
                    interpretation = describe_cluster(row)

                    st.markdown(
                        f"""
                        <div class="insight-card">
                            <strong>Cluster {cluster_id}: {cluster_name}</strong><br>
                            {interpretation}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.subheader("PCA View of Hierarchical Clusters")

                # This PCA plot provides a second visual check of the
                # hierarchical clustering output.
                pca_df, pca_model = create_pca_dataframe(X_processed, labels)
                pca_fig = create_pca_scatter(
                    pca_df,
                    "Hierarchical Clusters Projected with PCA",
                )

                st.plotly_chart(pca_fig, use_container_width=True)

                with st.expander("Why view hierarchical clusters with PCA?"):
                    st.write(
                        """
                        The dendrogram shows the tree structure of the clusters, while the
                        PCA plot shows those same cluster assignments in a two-dimensional
                        space. This makes it easier to compare whether the hierarchical
                        clusters also appear visually separated.

                        If groups are separated in the PCA plot, that supports the idea
                        that the selected features are capturing meaningful behavioral
                        differences between investors.
                        """
                    )

                # Export the dataset with hierarchical cluster labels for
                # additional analysis or reporting.
                st.download_button(
                    label="Download hierarchical clustered dataset",
                    data=clustered_df.to_csv(index=False).encode("utf-8"),
                    file_name="investor_persona_hierarchical_clusters.csv",
                    mime="text/csv",
                )
