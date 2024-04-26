import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
from sklearn.impute import SimpleImputer

# Load the Dataset
dataset = pd.read_csv('combined_dataset.csv')

print("Dataset Shape:", dataset.shape)
print("First Few Rows:")
print(dataset.head())
print("Summary Statistics:")
print(dataset.describe())
print("Missing Values:")
print(dataset.isnull().sum())

# Drop unnecessary columns
columns_to_drop = ["Src IP Addr", "Dst IP Addr", "Bytes", "Date first seen", "attackType", "attackID", "attackDescription"]
data = data.drop(columns=columns_to_drop)

# Encoding categorical var
label_encoder = LabelEncoder()
data["Proto"] = label_encoder.fit_transform(data["Proto"])
data["Flags"] = label_encoder.fit_transform(data["Flags"])
data["label"] = label_encoder.fit_transform(data["label"])

# Spliting
X = data.drop(columns=["label"])
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Check for missing values in the dataset
print("No. of missing values in X_train_scaled:", np.isnan(X_train_scaled).sum())

# Impute missing values using mean imputation
imputer = SimpleImputer(strategy="mean")
X_train_scaled_imputed = imputer.fit_transform(X_train_scaled)
X_test_scaled_imputed = imputer.transform(X_test_scaled)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Train Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled_imputed, y_train)
rf_pred = rf.predict(X_test_scaled_imputed)
rf_accuracy = accuracy_score(y_test, rf_pred)
print("RF Accuracy:", rf_accuracy)
print("Classification Report for RF:\n", classification_report(y_test, rf_pred))

# Plot confusion matrix for Random Forest
rf_cm = confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(5, 2.5))
sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix - Random Forest")
plt.show()

# Train Gaussian Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(X_train_scaled_imputed, y_train)
gnb_pred = gnb.predict(X_test_scaled_imputed)
gnb_accuracy = accuracy_score(y_test, gnb_pred)
print("Gaussian Naive Bayes Accuracy:", gnb_accuracy)
print("Classification Report for Gaussian Naive Bayes:\n", classification_report(y_test, gnb_pred))

# Plot confusion matrix for Gaussian Naive Bayes
gnb_cm = confusion_matrix(y_test, gnb_pred)
plt.figure(figsize=(5, 2.5))
sns.heatmap(gnb_cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix - Gaussian Naive Bayes")
plt.show()

import lightgbm as lgb
# Train LightGBM classifier
lgb_classifier = lgb.LGBMClassifier()
lgb_classifier.fit(X_train_scaled_imputed, y_train)
lgb_pred = lgb_classifier.predict(X_test_scaled_imputed)
lgb_accuracy = accuracy_score(y_test, lgb_pred)
print("LightGBM Accuracy:", lgb_accuracy)
print("Classification Report for LightGBM:\n", classification_report(y_test, lgb_pred))

# Plot confusion matrix for LightGBM
lgb_cm = confusion_matrix(y_test, lgb_pred)
plt.figure(figsize=(5, 2.5))
sns.heatmap(lgb_cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix - LightGBM")
plt.show()

# Initialize KMeans with the desired number of clusters (e.g., 3)
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit KMeans to the dataset
kmeans.fit(X_train_scaled)

# Obtain cluster labels for each data point
cluster_labels = kmeans.labels_

# Print cluster labels
print("Cluster labels:", cluster_labels)

# Scatter plot of the clustered data
plt.figure(figsize=(8, 6))
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.5)
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')
plt.show()

