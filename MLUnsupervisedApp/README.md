# Investor Persona Clustering App – Unsupervised Machine Learning

## Project Overview

**Investor Persona Clustering App** is an interactive unsupervised machine learning web application built with Streamlit that uncovers hidden investor behavior profiles based on portfolio characteristics and trading patterns.

Unlike traditional supervised models that predict predefined labels, this app focuses on discovering structure within the data. By applying clustering techniques such as K-Means and hierarchical clustering, along with dimensionality reduction through Principal Component Analysis (PCA), the app identifies distinct investor personas without prior labeling.

Users can upload their own dataset or explore a built-in simulated investor dataset, adjust model parameters, and visually interpret how investors group based on behavior, diversification, and risk exposure.

---

## App Preview

### Overview Dashboard
![Overview Screenshot](images/overview.png)

This view provides a high-level summary of the dataset, including the total number of rows, columns, numeric features, and default modeling features. It also displays a preview of the investor dataset alongside a summary of app capabilities, helping users quickly understand both the data structure and available analysis tools.

---

### Data Exploration
![Explore Data Screenshot](images/explore.png)

The data exploration page visualizes the distribution of selected features using histograms. In this example, volatility exposure shows clear variation across investors, helping users identify patterns, skewness, and potential groupings before applying clustering models.

---

### K-Means Clustering Results
![Clustering Screenshot](images/clustering.png)

This visualization displays K-Means clustering results projected into two dimensions using PCA. Each point represents an investor, and color-coded clusters reveal distinct behavioral groupings. The separation between clusters demonstrates how effectively the model differentiates investor personas based on portfolio characteristics.

---

### PCA Visualization
![PCA Screenshot](images/pca.png)

This PCA scatter plot shows the two-dimensional projection of the dataset after dimensionality reduction. The axes (PC1 and PC2) capture the majority of variance in the data, allowing complex investor behavior patterns to be visualized in a simplified space where similar investors appear closer together.

---

### Hierarchical Clustering (Dendrogram)
![Hierarchical Screenshot](images/hierarchical.png)

The dendrogram illustrates how investors are grouped through hierarchical clustering. Each branch represents a merge between clusters, with height indicating distance (dissimilarity). This view reveals the nested structure of investor relationships and how clusters form at different similarity thresholds.

---

## Key Features

### Dataset Flexibility
- Upload custom tabular datasets (CSV or Excel)
- Use a built-in investor behavior dataset with realistic variation and noise
- Download sample dataset for experimentation

### Explore Data
- Dataset preview and summary statistics
- Feature distribution visualization
- Correlation matrix heatmap
- Automatic detection of numeric features

### K-Means Clustering
- Adjustable number of clusters (k)
- Feature selection for clustering
- Optional feature standardization
- Model evaluation with:
  - Silhouette score
  - Elbow plot
- Cluster summary tables (mean feature values)
- Plain-English cluster interpretations (investor personas)

### PCA (Principal Component Analysis)
- Dimensionality reduction to 2 components
- Interactive 2D scatterplot visualization
- Explained variance metrics
- Feature loadings for interpretability

### Hierarchical Clustering
- Agglomerative clustering (Ward linkage)
- Adjustable number of clusters
- Dendrogram visualization
- PCA-based cluster visualization
- Cluster summary and comparison

### Export Functionality
- Download clustered datasets (K-Means and hierarchical)
- Enables further analysis outside the app

---

## Technical Stack

| Component | Technology | Purpose |
|----------|------------|--------|
| Web Framework | Streamlit | Interactive UI |
| Data Processing | Pandas, NumPy | Data manipulation |
| Machine Learning | scikit-learn | Clustering & PCA |
| Visualization | Plotly, Matplotlib | Charts |
| Clustering Tools | SciPy | Hierarchical clustering |
| Navigation | streamlit-option-menu | Sidebar UI |

---

## Installation & Setup

### Run Locally

```bash
git clone https://github.com/passionhood/Hood-Data-Science-Portfolio.git
cd Hood-Data-Science-Portfolio/MLUnsupervisedApp
pip install -r requirements.txt
streamlit run main.py