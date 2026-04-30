# Investor Persona Clustering App
# Author: Passion Hood
# Course: MDSC 20009: Machine Learning for Data Science
# Description: This Streamlit app demonstrates unsupervised machine learning by allowing users to cluster investor behavior data using K-Means, Hierarchical Clustering, and PCA.
# Main features:
# - Upload a custom CSV or Excel dataset
# - Use a built-in simulated investor persona dataset
# - Select numeric features for clustering
# - Adjust the number of clusters interactively
# - View PCA visualizations, elbow plots, silhouette scores, dendrograms, and cluster summaries
# - Download the final dataset with assigned cluster labels

import io

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# Page configuration
st.set_page_config(
    page_title="Investor Persona Clustering App",
    page_icon="📊",
    layout="wide"
)

# Helper function: create a sample investor dataset
@st.cache_data
def generate_sample_dataset(seed: int = 42) -> pd.DataFrame:
    """
    Generates a simulated investor behavior dataset.

    The dataset includes five broad investor archetypes with realistic variation,
    plus a few noisy observations to make the clustering task less perfect and
    more reflective of real-world data.
    """
    np.random.seed(seed)
    rows = []

    def add_group(
        n,
        persona,
        turnover,
        holding_period,
        volatility,
        diversification,
        sector_concentration,
        holdings,
        cash,
        international,
        noise=0.05,
    ):
        """Adds simulated investors around a central persona profile."""
        for _ in range(n):
            rows.append({
                "investor_id": len(rows) + 1,
                "true_persona": persona,
                "portfolio_turnover": np.clip(np.random.normal(turnover, noise), 0, 1),
                "avg_holding_period_days": max(1, int(np.random.normal(holding_period, holding_period * 0.10))),
                "volatility_exposure": np.clip(np.random.normal(volatility, noise), 0, 1),
                "diversification_score": np.clip(np.random.normal(diversification, noise), 0, 1),
                "sector_concentration": np.clip(np.random.normal(sector_concentration, noise), 0, 1),
                "num_holdings": max(1, int(np.random.normal(holdings, 3))),
                "cash_allocation_pct": np.clip(np.random.normal(cash, noise), 0, 1),
                "international_exposure_pct": np.clip(np.random.normal(international, noise), 0, 1),
            })

    # Core investor groups intentionally designed for meaningful clustering.
    add_group(10, "Aggressive Trader", 0.85, 30, 0.90, 0.30, 0.80, 8, 0.05, 0.20)
    add_group(10, "Passive Long-Term Investor", 0.10, 380, 0.20, 0.85, 0.20, 30, 0.18, 0.45)
    add_group(10, "Balanced Investor", 0.50, 180, 0.50, 0.60, 0.50, 20, 0.10, 0.30)
    add_group(10, "Concentrated Growth Investor", 0.70, 90, 0.85, 0.40, 0.90, 10, 0.05, 0.20)
    add_group(10, "Conservative Wealth Preserver", 0.18, 320, 0.22, 0.75, 0.28, 27, 0.35, 0.38)

    # Add a few noisy rows to represent atypical investor behavior.
    for _ in range(5):
        rows.append({
            "investor_id": len(rows) + 1,
            "true_persona": "Noise / Outlier",
            "portfolio_turnover": np.random.uniform(0, 1),
            "avg_holding_period_days": int(np.random.uniform(10, 500)),
            "volatility_exposure": np.random.uniform(0, 1),
            "diversification_score": np.random.uniform(0, 1),
            "sector_concentration": np.random.uniform(0, 1),
            "num_holdings": int(np.random.uniform(5, 40)),
            "cash_allocation_pct": np.random.uniform(0, 1),
            "international_exposure_pct": np.random.uniform(0, 1),
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Helper function: load uploaded data
# -----------------------------------------------------------------------------
def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Reads a user-uploaded CSV or Excel file into a DataFrame."""
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if file_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)

    raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")


# -----------------------------------------------------------------------------
# Helper function: recommend a persona name based on cluster averages
# -----------------------------------------------------------------------------
def assign_persona_name(row: pd.Series) -> str:
    """
    Assigns a plain-English investor persona using cluster-level feature averages.

    This is not part of the machine learning model itself. It is an interpretation
    layer that helps users understand what each cluster likely represents.
    """
    turnover = row.get("portfolio_turnover", 0)
    holding = row.get("avg_holding_period_days", 0)
    volatility = row.get("volatility_exposure", 0)
    diversification = row.get("diversification_score", 0)
    sector = row.get("sector_concentration", 0)
    cash = row.get("cash_allocation_pct", 0)

    if turnover >= 0.65 and holding <= 120 and volatility >= 0.70:
        return "Aggressive Traders"
    if sector >= 0.70 and volatility >= 0.70 and diversification <= 0.50:
        return "Concentrated Growth Investors"
    if turnover <= 0.25 and holding >= 250 and diversification >= 0.70 and cash < 0.30:
        return "Passive Long-Term Investors"
    if cash >= 0.25 and volatility <= 0.40:
        return "Conservative Wealth Preservers"
    return "Balanced / Moderate Investors"


# -----------------------------------------------------------------------------
# App title and overview
# -----------------------------------------------------------------------------
st.title("Investor Persona Clustering App")

st.write(
    "This app uses unsupervised machine learning to uncover hidden investor "
    "profiles based on portfolio behavior, trading patterns, diversification, "
    "and risk exposure. Users can upload their own tabular dataset or explore a "
    "built-in simulated investor dataset."
)


# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
st.sidebar.header("Data and Model Settings")

# Let users choose between a built-in sample dataset and their own uploaded file.
data_source = st.sidebar.radio(
    "Choose a data source:",
    ["Use sample investor dataset", "Upload my own dataset"]
)

if data_source == "Upload my own dataset":
    uploaded_file = st.sidebar.file_uploader(
        "Upload a CSV or Excel file",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is not None:
        try:
            df = load_uploaded_file(uploaded_file)
        except Exception as e:
            st.error(f"There was a problem reading the file: {e}")
            st.stop()
    else:
        st.info("Upload a dataset to begin, or switch to the sample investor dataset.")
        st.stop()
else:
    df = generate_sample_dataset()

# Remove completely empty columns if a user uploads a messy spreadsheet.
df = df.dropna(axis=1, how="all")

if df.empty:
    st.error("The selected dataset is empty. Please upload a file with data.")
    st.stop()


# -----------------------------------------------------------------------------
# Dataset preview
# -----------------------------------------------------------------------------
st.subheader("1. Dataset Preview")

st.write(f"Rows: **{df.shape[0]}** | Columns: **{df.shape[1]}**")
st.dataframe(df.head(10), use_container_width=True)

# Provide sample data download when using the built-in dataset.
if data_source == "Use sample investor dataset":
    st.download_button(
        label="Download Sample Dataset",
        data=df.to_csv(index=False),
        file_name="investor_persona_dataset.csv",
        mime="text/csv"
    )


# -----------------------------------------------------------------------------
# Feature selection
# -----------------------------------------------------------------------------
st.subheader("2. Select Features for Clustering")

numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

# Exclude ID-like columns by default because they do not represent investor behavior.
default_features = [
    col for col in numeric_columns
    if col.lower() not in ["investor_id", "id", "cluster", "kmeans_cluster", "hierarchical_cluster"]
]

selected_features = st.multiselect(
    "Choose numeric columns to use for clustering:",
    options=numeric_columns,
    default=default_features
)

if len(selected_features) < 2:
    st.warning("Please select at least two numeric features for clustering.")
    st.stop()

# Keep only selected features and remove rows with missing values in those fields.
model_df = df[selected_features].dropna()

if model_df.shape[0] < 3:
    st.error("The selected features do not contain enough complete rows for clustering.")
    st.stop()

st.write(f"Complete rows available for modeling: **{model_df.shape[0]}**")


# -----------------------------------------------------------------------------
# Scaling and hyperparameter controls
# -----------------------------------------------------------------------------
st.subheader("3. Model Controls")

col1, col2, col3 = st.columns(3)

with col1:
    scaler_choice = st.selectbox(
        "Feature scaling method:",
        ["StandardScaler", "MinMaxScaler", "No scaling"]
    )

with col2:
    max_clusters = min(10, model_df.shape[0] - 1)
    n_clusters = st.slider(
        "Number of clusters:",
        min_value=2,
        max_value=max_clusters,
        value=min(5, max_clusters)
    )

with col3:
    linkage_method = st.selectbox(
        "Hierarchical linkage method:",
        ["ward", "complete", "average", "single"]
    )

# Scale selected features so that variables measured on different scales do not dominate the model.
if scaler_choice == "StandardScaler":
    scaler = StandardScaler()
elif scaler_choice == "MinMaxScaler":
    scaler = MinMaxScaler()
else:
    scaler = None

if scaler is not None:
    X_scaled = scaler.fit_transform(model_df)
else:
    X_scaled = model_df.values


# -----------------------------------------------------------------------------
# K-Means clustering
# -----------------------------------------------------------------------------
st.subheader("4. K-Means Clustering Results")

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Silhouette score measures how well-separated the clusters are.
sil_score = silhouette_score(X_scaled, kmeans_labels)

metric_col1, metric_col2 = st.columns(2)
metric_col1.metric("Selected Number of Clusters", n_clusters)
metric_col2.metric("Silhouette Score", f"{sil_score:.3f}")

st.write(
    "The silhouette score ranges from -1 to 1. Higher values generally indicate "
    "that observations are better matched to their assigned cluster and more "
    "separated from other clusters."
)


# -----------------------------------------------------------------------------
# PCA visualization
# -----------------------------------------------------------------------------
st.subheader("5. PCA Cluster Visualization")

pca = PCA(n_components=2, random_state=42)
pca_components = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame({
    "PC1": pca_components[:, 0],
    "PC2": pca_components[:, 1],
    "KMeans Cluster": kmeans_labels.astype(str)
})

explained_variance = pca.explained_variance_ratio_.sum()
st.write(
    f"The first two principal components explain **{explained_variance:.1%}** "
    "of the variation in the selected features."
)

fig, ax = plt.subplots(figsize=(8, 5))
for cluster in sorted(pca_df["KMeans Cluster"].unique()):
    cluster_points = pca_df[pca_df["KMeans Cluster"] == cluster]
    ax.scatter(cluster_points["PC1"], cluster_points["PC2"], label=f"Cluster {cluster}", alpha=0.75)

ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_title("Investor Clusters Visualized with PCA")
ax.legend()
st.pyplot(fig)


# -----------------------------------------------------------------------------
# Elbow plot
# -----------------------------------------------------------------------------
st.subheader("6. Elbow Plot")

inertias = []
k_values = range(2, max_clusters + 1)

for k in k_values:
    temp_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    temp_kmeans.fit(X_scaled)
    inertias.append(temp_kmeans.inertia_)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(list(k_values), inertias, marker="o")
ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("Inertia")
ax.set_title("Elbow Plot for K-Means Clustering")
st.pyplot(fig)

st.write(
    "The elbow plot helps users evaluate how the number of clusters affects model "
    "fit. A useful k value often appears near the point where inertia begins to "
    "decrease more slowly."
)


# -----------------------------------------------------------------------------
# Hierarchical clustering and dendrogram
# -----------------------------------------------------------------------------
st.subheader("7. Hierarchical Clustering Dendrogram")

# Ward linkage works best with Euclidean distances and is commonly used for compact clusters.
linked = linkage(X_scaled, method=linkage_method)

fig, ax = plt.subplots(figsize=(10, 5))
dendrogram(linked, ax=ax, truncate_mode="lastp", p=20)
ax.set_title("Hierarchical Clustering Dendrogram")
ax.set_xlabel("Investors or Clustered Groups")
ax.set_ylabel("Distance")
st.pyplot(fig)

hierarchical_model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
hierarchical_labels = hierarchical_model.fit_predict(X_scaled)


# -----------------------------------------------------------------------------
# Cluster interpretation and summary
# -----------------------------------------------------------------------------
st.subheader("8. Investor Persona Insights")

# Reattach labels to the original complete rows used for modeling.
results_df = df.loc[model_df.index].copy()
results_df["kmeans_cluster"] = kmeans_labels
results_df["hierarchical_cluster"] = hierarchical_labels
results_df["PC1"] = pca_components[:, 0]
results_df["PC2"] = pca_components[:, 1]

# Summarize clusters by the average selected feature values.
cluster_summary = results_df.groupby("kmeans_cluster")[selected_features].mean().round(3)
cluster_summary["suggested_persona"] = cluster_summary.apply(assign_persona_name, axis=1)
cluster_summary["investor_count"] = results_df.groupby("kmeans_cluster").size()

st.write("Cluster labels are assigned by the algorithm and do not imply ranking or order.")
st.dataframe(cluster_summary, use_container_width=True)

st.write("Dataset with assigned cluster labels:")
st.dataframe(results_df, use_container_width=True)

# Download clustered results
st.subheader("9. Download Results")

csv_buffer = io.StringIO()
results_df.to_csv(csv_buffer, index=False)

st.download_button(
    label="Download Clustered Dataset",
    data=csv_buffer.getvalue(),
    file_name="investor_persona_clustered_results.csv",
    mime="text/csv"
)

# Footer explanation
st.caption(
    "Built with Streamlit, pandas, scikit-learn, scipy, and matplotlib. "
    "This project demonstrates unsupervised machine learning through clustering, "
    "dimensionality reduction, model evaluation, and interactive data exploration."
)
