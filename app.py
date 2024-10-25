import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the Iris dataset (no labels are used for unsupervised learning)
data = load_iris()
X = data.data

# Split the dataset (optional for unsupervised learning)
X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

# Define a pipeline with PCA for dimensionality reduction and K-Means for clustering
pipeline = Pipeline([
    ('scaler', StandardScaler()),    # Standardizes the data
    ('pca', PCA(n_components=2)),    # Reduces to 2 dimensions for visualization
    ('kmeans', KMeans(n_clusters=3, random_state=42))  # Clustering into 3 clusters
])

# Fit the pipeline to the training data
pipeline.fit(X_train)

# Get the cluster labels
cluster_labels = pipeline.named_steps['kmeans'].labels_

# Predict cluster labels for the test data
cluster_labels_test = pipeline.predict(X_test)

# Calculate the silhouette score to evaluate clustering performance
silhouette_avg = silhouette_score(X_train, cluster_labels)

print("Silhouette Score on training data:", silhouette_avg)

# Print the PCA explained variance ratio
print("Explained Variance Ratio by PCA:", pipeline.named_steps['pca'].explained_variance_ratio_)

# Show cluster labels for the test data
print("Cluster labels for the test data:", cluster_labels_test)
