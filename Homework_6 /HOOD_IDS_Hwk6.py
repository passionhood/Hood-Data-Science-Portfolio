# Part 1: Data Loading and Exploration
import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing 

#Loading the dataset from sklearn
housing = fetch_california_housing()

#Creating a Pandas DataFrame for the features and a Series for the target variable
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name="med_house_value")

print("X shape:", X.shape)
print("y shape:", y.shape)

#displaying the first five rows of the dataset
print("First 5 rows of X:")
print(X.head())
print("\nFirst 5 values of y:")
print(y.head())

# printing feature names and checking for missing values
print("Feature names:")
print(list(X.columns))
print("\nMissing values per column:")
print(X.isna().sum())
print("\nMissing values in target:")
print(y.isna().sum())

#generate summary statstics 
print("Summary statistics for features:")
print(X.describe())

print("\nSummary statistics for target:")
print(y.describe())

# Part 2: Linear Regression on Unscaled Data









# Part 3: Feature Selection and Simplified Model