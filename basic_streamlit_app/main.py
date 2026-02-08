import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Always show something immediately
st.title("📊 Cryptocurrency Dashboard")
st.write("✅ App started (if you see this, Streamlit is running this file).")

# -------------------------
# Load the data (robust)
# -------------------------
# Try both possible filenames
possible_paths = [
    Path(__file__).parent / "CryptocurrencyData copy.csv",
    Path(__file__).parent / "CryptocurrencyData.csv",
    Path(__file__).parent / "data" / "CryptocurrencyData copy.csv",
    Path(__file__).parent / "data" / "CryptocurrencyData.csv",
]

csv_path = None
for path in possible_paths:
    if path.exists():
        csv_path = path
        break

if csv_path is None:
    # List files in current directory for debugging
    st.error("❌ Could not find CSV file")
    st.write("Files in current directory:", list(Path(__file__).parent.glob("*.csv")))
    st.write("Files in data directory:", list(Path(__file__).parent.glob("data/*.csv")))
    st.stop()

st.write(f"✅ Using CSV from: {csv_path}")

try:
    df = pd.read_csv(csv_path)
    st.write(f"✅ CSV loaded! Shape: {df.shape}")
except Exception as e:
    st.error("❌ Could not load the CSV file.")
    st.exception(e)
    st.stop()

# Clean up column names
df.columns = df.columns.str.strip()
st.write("Columns found:", df.columns.tolist())

# -------------------------
# Clean currency columns
# -------------------------
def to_number(x):
    """Convert currency strings to float"""
    if pd.isna(x):
        return float("nan")
    if isinstance(x, str):
        # Remove common currency/formatting characters
        x = x.strip().replace("$", "").replace(",", "").replace(" ", "")
        if x in ("", "-", "N/A", "NA", "∞"):
            return float("nan")
    try:
        return float(x)
    except:
        return float("nan")

# Apply cleaning to numeric columns
numeric_cols = ["Price", "1h", "24h", "7d", "30d", "24h Volume", "Market Cap"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].apply(to_number)

# -------------------------
# App description + data preview
# -------------------------
st.write("Pick a coin and filter by market cap to explore the dataset.")
st.subheader("Sample of the data")
st.dataframe(df.head(10), use_container_width=True)

# -------------------------
# Filters (sidebar)
# -------------------------
st.sidebar.header("Filters")

if "Coin Name" not in df.columns:
    st.error("❌ Your CSV is missing the 'Coin Name' column.")
    st.write("Columns found:", df.columns.tolist())
    st.stop()

coin_names = sorted(df["Coin Name"].dropna().unique().tolist())

if not coin_names:
    st.error("❌ No coin names found in the data")
    st.stop()

coin = st.sidebar.selectbox("Choose a cryptocurrency:", coin_names)

filtered_df = df[df["Coin Name"] == coin].copy()

# Market Cap filter
if "Market Cap" in df.columns:
    cap_values = df["Market Cap"].dropna()
    if len(cap_values) > 0:
        min_cap = float(cap_values.min())
        max_cap = float(cap_values.max())

        cap_range = st.sidebar.slider(
            "Market Cap range:",
            min_value=min_cap,
            max_value=max_cap,
            value=(min_cap, max_cap),
        )

        filtered_df = filtered_df[
            (filtered_df["Market Cap"] >= cap_range[0]) 
            & (filtered_df["Market Cap"] <= cap_range[1])
        ]
    else:
        st.warning("No valid Market Cap values found, so the slider filter is unavailable.")
else:
    st.warning("Market Cap column not found")

st.subheader("Filtered results")
if len(filtered_df) > 0:
    st.dataframe(filtered_df, use_container_width=True)
else:
    st.info("No results match the selected filters")

# -------------------------
# Simple chart
# -------------------------
st.subheader("Top 10 coins by Market Cap")

if "Market Cap" in df.columns:
    top_10 = df.dropna(subset=["Market Cap", "Coin Name"]).nlargest(10, "Market Cap")
    
    if len(top_10) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_10["Coin Name"], top_10["Market Cap"] / 1e9)  # Convert to billions for readability
        ax.set_xlabel("Market Cap (Billions $)")
        ax.set_ylabel("Coin")
        ax.set_title("Top 10 Cryptocurrencies by Market Cap")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Not enough Market Cap data to plot top 10.")
else:
    st.error("Market Cap column not found")
