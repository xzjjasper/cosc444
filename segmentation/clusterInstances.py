import os
import numpy as np
from sklearn.cluster import KMeans

# === CONFIG ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Clothes descriptor
# IN_DIR = os.path.join(BASE_DIR, "output", "cluster_clothes")
# Body size descriptor
IN_DIR = os.path.join(BASE_DIR, "output", "cluster_body_size")
OUT_PATH = os.path.join(BASE_DIR, "output", "cluster_labels_bodySize.npy")

# === Load extracted descriptors ===
features = np.load(os.path.join(IN_DIR, "features.npy"))
identifiers = np.load(os.path.join(IN_DIR, "identifiers.npy"))

# === Choose number of clusters ===
n_clusters = 7  # You can tune this based on expected variety

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(features)

# === Save as [instance_id, cluster_id] pairs ===
cluster_data = np.array(list(zip(identifiers, labels)), dtype=object)
np.save(OUT_PATH, cluster_data)

# === Print summary ===
print(f"Saved {len(cluster_data)} cluster assignments to: {OUT_PATH}")
unique, counts = np.unique(labels, return_counts=True)
dist = dict(zip(unique, counts))
print("Cluster distribution:", dist)
