# ==============================
# 📊 LONDON CRIME DATA - EDA
# ==============================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# ------------------------------
# 1. Load Data
# ------------------------------

train_data = pd.read_csv('train.csv', header=None)
train_labels = pd.read_csv('trainLabels.csv', header=None)
test_data = pd.read_csv('test.csv', header=None)

# ------------------------------
# 2. Basic Info
# ------------------------------

print("Train data shape:", train_data.shape)
print("Train labels shape:", train_labels.shape)
print("\nTrain data preview:\n", train_data.head())
print("\nTrain labels preview:\n", train_labels.head())

print("\nMissing values in train data:\n", train_data.isnull().sum())
print("Missing values in train labels:\n", train_labels.isnull().sum())

# ------------------------------
# 3. Label Distribution
# ------------------------------

# Bar Plot
train_labels.value_counts().plot(kind='bar', color='mediumseagreen')
plt.title('Label Distribution')
plt.xlabel('Crime Type')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Pie Chart (Interactive)
label_dist = train_labels.value_counts().reset_index()
label_dist.columns = ['Crime Type', 'Count']

fig = px.pie(
    label_dist,
    values='Count',
    names='Crime Type',
    hole=0.6,
    opacity=0.9,
    title="London Crime Type Distribution"
)
fig.update_traces(textinfo='percent+label')
fig.show()

# ------------------------------
# 4. Correlation Matrix
# ------------------------------

plt.figure(figsize=(12, 10))
sns.heatmap(train_data.corr(), cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()

# ------------------------------
# 5. Correlation with Target
# ------------------------------

correlation_with_target = train_data.corrwith(train_labels[0])
print("\n📈 Correlation of Each Feature with the Target:\n")
print(correlation_with_target.sort_values(ascending=False))

# ------------------------------
# 6. Test Data Overview
# ------------------------------

print("\nTest Data Info:")
test_data.info()
