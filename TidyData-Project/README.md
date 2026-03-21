# TidyData-Project

# Federal R&D Budget Analysis Using Tidy Data Principles
## Overview
This project focuses on cleaning, restructuring, and analyzing a Federal Research & Development (R&D) budget dataset using **tidy data principles**. The original dataset was in a wide format, which limited its usability for analysis and visualization.

By transforming the dataset into a tidy format, this project enables more efficient data manipulation and clearer insights into trends in federal R&D spending across departments and over time.

## Objectives

* Apply tidy data principles to restructure a real-world dataset
* Clean and prepare data for analysis using Pandas
* Perform exploratory data analysis (EDA)
* Develop clear and informative visualizations

## Tidy Data Implementation

Tidy data requires:
* Each variable to be in its own column
* Each observation to be in its own row
* Each type of observational unit to form its own table

### Application in This Project:
* Converted wide-format yearly columns into a single **Year** variable using `melt()`
* Split combined variables (e.g., Year and GDP) into separate columns using string operations
* Cleaned column values using `str.replace()`
* Ensured each row represents a unique **department–year observation**
* Converted budget values into a numeric format for analysis

## Data Cleaning and Transformation
The dataset was processed through the following steps:
1. **Data Loading**

   * Imported the dataset into a Pandas DataFrame
   * Inspected structure and identified missing values

2. **Reshaping**

   * Transformed data from wide to long format using `melt()`

3. **String Processing**

   * Extracted and separated variables using `str.split()`
   * Cleaned text data using `str.replace()`

4. **Data Preparation**

   * Converted budget values to numeric
   * Removed or handled missing values

## Exploratory Data Analysis
### Summary Statistics
Descriptive statistics were generated to understand the distribution and scale of federal R&D spending.

### Pivot Table Analysis
A pivot table was used to examine budget allocation across departments and years:

```python
df_melted.pivot_table(
    values="Budget",
    index="department",
    columns="Year",
    aggfunc="sum"
)
```
## Visualizations
### Total Federal R&D Spending Over Time
A line chart illustrating overall trends in government R&D expenditures.

### Departmental Budget Comparison (Most Recent Year)
A horizontal bar chart comparing departmental R&D budgets for the latest available year.


## How to Run the Project
### Requirements

Install required libraries:

```bash
pip install pandas matplotlib seaborn
```
### Execution Steps

1. Open the Jupyter Notebook: `Tidy_Data_Fed_RD.ipynb`
2. Run all cells sequentially
3. Review outputs including cleaned data, summary statistics, and visualizations

## Dataset Description

* **Source:** Federal R&D Budgets dataset
* **Structure:**

  * Rows represent government departments
  * Columns represent yearly budgets (with embedded GDP values)

### Preprocessing

* Converted dataset from wide to long format
* Extracted and cleaned year-related information
* Standardized budget values for analysis

## Significance of Tidy Data

Applying tidy data principles improves:

* Analytical efficiency
* Code readability and reproducibility
* Compatibility with visualization and statistical tools

This transformation enables more meaningful analysis of federal spending trends.

## Portfolio Relevance

This project demonstrates core competencies in **data cleaning, transformation, and visualization**, which are foundational for data-driven decision-making in finance and analytics. It reflects the ability to work with real-world datasets and extract actionable insights through structured analysis.


## References

* Pandas Cheat Sheet: https://www.geeksforgeeks.org/pandas-cheat-sheet/
* Wickham, H. (2014). *Tidy Data*: https://vita.had.co.nz/papers/tidy-data.pdf