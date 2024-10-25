import pytest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Fixture for data preparation
@pytest.fixture
def data():
    data = load_iris()
    X = data.data
    return X

# Test the unsupervised learning pipeline
def test_unsupervised_pipeline(data):
    X_train = data

    # Define a pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=2)),
        ('kmeans', KMeans(n_clusters=3, random_state=42))
    ])

    # Fit the pipeline
    pipeline.fit(X_train)

    # Get the cluster labels
    cluster_labels = pipeline.named_steps['kmeans'].labels_

    # Compute the silhouette score
    silhouette_avg = silhouette_score(X_train, cluster_labels)
    
    # Assert that the silhouette score is reasonable
    assert silhouette_avg > 0, "Silhouette score should be positive."

    # Assert that PCA has reduced the data to 2 components
    assert pipeline.named_steps['pca'].n_components == 2, "PCA should reduce to 2 components."

    print("Silhouette Score:", silhouette_avg)

# Run the tests using pytest
if __name__ == "__main__":
    pytest.main()
