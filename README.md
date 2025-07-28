# week-7-python
Task 1: Load and Explore the Dataset
Choose a dataset in CSV format (for example, you can use datasets like the Iris dataset, a sales dataset, or any dataset of your choice).
Load the dataset using pandas.
Display the first few rows of the dataset using .head() to inspect the data.
Explore the structure of the dataset by checking the data types and any missing values.
Clean the dataset by either filling or dropping any missing values.
Task 2: Basic Data Analysis
Compute the basic statistics of the numerical columns (e.g., mean, median, standard deviation) using .describe().
Perform groupings on a categorical column (for example, species, region, or department) and compute the mean of a numerical column for each group.
Identify any patterns or interesting findings from your analysis.
Task 3: Data Visualization
Create at least four different types of visualizations:
Line chart showing trends over time (for example, a time-series of sales data).
Bar chart showing the comparison of a numerical value across categories (e.g., average petal length per species).
Histogram of a numerical column to understand its distribution.
Scatter plot to visualize the relationship between two numerical columns (e.g., sepal length vs. petal length).
Customize your plots with titles, labels for axes, and legends where necessary.



Additional Instructions

Dataset Suggestions:

You can use publicly available datasets from sites like Kaggle or UCI Machine Learning Repository.
The Iris dataset (a classic dataset for classification problems) can be accessed via sklearn.datasets.load_iris(), which can be used for the analysis.

Plot Customization:

Customize the plots using the matplotlib library to add titles, axis labels, and legends.
Use seaborn for additional plotting styles, which can make your charts more visually appealing.

Error Handling:

Handle possible errors during the file reading (e.g., file not found), missing data, or incorrect data types by using exception-handling mechanisms (try, except).



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Task 1: Load and Explore the Dataset ---

# 1.1 Choose a dataset in CSV format
# For demonstration, we'll create a synthetic sales dataset.
# In a real scenario, you would replace this with loading your actual CSV file.

# Define data for a synthetic sales dataset
data = {
    'Date': pd.to_datetime(pd.date_range(start='2023-01-01', periods=100, freq='D')),
    'Region': np.random.choice(['North', 'South', 'East', 'West'], 100),
    'Product': np.random.choice(['A', 'B', 'C', 'D'], 100),
    'Sales': np.random.randint(50, 500, 100),
    'Quantity': np.random.randint(1, 20, 100),
    'Profit': np.random.uniform(10, 100, 100)
}
df = pd.DataFrame(data)

# Introduce some missing values for demonstration
df.loc[10:15, 'Sales'] = np.nan
df.loc[20, 'Region'] = np.nan
df.loc[30:32, 'Profit'] = np.nan

# Save the synthetic dataset to a CSV file
csv_file_path = 'synthetic_sales_data.csv'
try:
    df.to_csv(csv_file_path, index=False)
    print(f"Synthetic dataset saved to {csv_file_path}")
except IOError as e:
    print(f"Error saving CSV file: {e}")
    # If saving fails, we might still proceed with the DataFrame in memory
    # but the user might not be able to find the file later.

# 1.2 Load the dataset using pandas
print("\n--- Task 1: Load and Explore the Dataset ---")
try:
    df = pd.read_csv(csv_file_path)
    print(f"Dataset '{csv_file_path}' loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file '{csv_file_path}' was not found.")
    print("Please ensure the CSV file exists in the same directory as the script.")
    exit() # Exit if the file cannot be loaded
except pd.errors.EmptyDataError:
    print(f"Error: The file '{csv_file_path}' is empty.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading the file: {e}")
    exit()

# Convert 'Date' column to datetime objects
df['Date'] = pd.to_datetime(df['Date'])

# 1.3 Display the first few rows of the dataset
print("\nFirst 5 rows of the dataset:")
print(df.head())

# 1.4 Explore the structure of the dataset
print("\nDataset Information (Data Types and Non-Null Counts):")
df.info()

print("\nMissing values before cleaning:")
print(df.isnull().sum())

# 1.5 Clean the dataset by either filling or dropping any missing values
# For numerical columns, we'll fill missing values with the median.
# For categorical columns, we'll drop rows with missing values as they might be critical.
for col in df.columns:
    if df[col].isnull().any():
        if pd.api.types.is_numeric_dtype(df[col]):
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"Filled missing values in '{col}' with median: {median_val}")
        else: # Assuming categorical or other types where dropping is safer
            df.dropna(subset=[col], inplace=True)
            print(f"Dropped rows with missing values in '{col}'")

print("\nMissing values after cleaning:")
print(df.isnull().sum())
print("\nDataset info after cleaning:")
df.info()


# --- Task 2: Basic Data Analysis ---
print("\n--- Task 2: Basic Data Analysis ---")

# 2.1 Compute the basic statistics of the numerical columns
print("\nBasic statistics of numerical columns:")
print(df.describe())

# 2.2 Perform groupings on a categorical column and compute the mean of a numerical column for each group
# Example: Mean Sales per Region
print("\nMean Sales per Region:")
mean_sales_per_region = df.groupby('Region')['Sales'].mean().sort_values(ascending=False)
print(mean_sales_per_region)

# Example: Mean Profit per Product
print("\nMean Profit per Product:")
mean_profit_per_product = df.groupby('Product')['Profit'].mean().sort_values(ascending=False)
print(mean_profit_per_product)

# 2.3 Identify any patterns or interesting findings from your analysis
print("\n--- Patterns and Findings ---")
print("From the analysis:")
print(f"- The average sales per region vary, with '{mean_sales_per_region.index[0]}' having the highest average sales.")
print(f"- Product '{mean_profit_per_product.index[0]}' appears to be the most profitable on average.")
print("- The distribution of 'Sales' (from .describe()) shows a range from min to max, with the mean indicating the central tendency.")


# --- Task 3: Data Visualization ---
print("\n--- Task 3: Data Visualization ---")

# Set a style for seaborn plots for better aesthetics
sns.set_style("whitegrid")
plt.figure(figsize=(15, 12))

# 3.1 Line chart showing trends over time (e.g., daily sales)
plt.subplot(2, 2, 1) # 2 rows, 2 columns, 1st subplot
daily_sales = df.groupby('Date')['Sales'].sum()
plt.plot(daily_sales.index, daily_sales.values, marker='o', linestyle='-', color='skyblue')
plt.title('Daily Sales Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# 3.2 Bar chart showing the comparison of a numerical value across categories
plt.subplot(2, 2, 2) # 2 rows, 2 columns, 2nd subplot
sns.barplot(x=mean_sales_per_region.index, y=mean_sales_per_region.values, palette='viridis')
plt.title('Average Sales per Region')
plt.xlabel('Region')
plt.ylabel('Average Sales')
plt.tight_layout()

# 3.3 Histogram of a numerical column to understand its distribution
plt.subplot(2, 2, 3) # 2 rows, 2 columns, 3rd subplot
sns.histplot(df['Sales'], bins=10, kde=True, color='lightcoral')
plt.title('Distribution of Sales')
plt.xlabel('Sales Amount')
plt.ylabel('Frequency')
plt.tight_layout()

# 3.4 Scatter plot to visualize the relationship between two numerical columns
plt.subplot(2, 2, 4) # 2 rows, 2 columns, 4th subplot
sns.scatterplot(x=df['Quantity'], y=df['Sales'], hue=df['Region'], palette='plasma', s=100, alpha=0.8)
plt.title('Sales vs. Quantity (by Region)')
plt.xlabel('Quantity Sold')
plt.ylabel('Sales Amount')
plt.legend(title='Region')
plt.tight_layout()

# Display all plots
plt.show()

print("\nData analysis and visualization complete. Plots displayed.")

# Clean up the generated CSV file
try:
    if os.path.exists(csv_file_path):
        os.remove(csv_file_path)
        print(f"Cleaned up: Removed '{csv_file_path}'")
except Exception as e:
    print(f"Error removing temporary CSV file: {e}")
