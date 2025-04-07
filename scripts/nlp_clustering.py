import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# Load the tokenized data
df = pd.read_csv('../data/processed/tokenized_data_example.csv')
print("read the output from the previous step")

# Generate document embeddings using a sentence-transformers model
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['tokenized_text'].tolist(), show_progress_bar=True)
print("Embeddings generated.")

# Determine the optimal number of clusters using the elbow method
inertias = []
K_range = range(2, 15)  # Try cluster numbers from 2 to 14
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(embeddings)
    inertias.append(kmeans.inertia_)

# Plot the elbow curve (this is interactive; close the plot to proceed)
plt.figure(figsize=(8, 4))
plt.plot(K_range, inertias, marker='o')
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia (Sum of Squared Distances)")
plt.title("Elbow Method to Determine Optimal K")
plt.show()

# adjust optimal K value based on what you see in the plot.
optimal_k = 5
print("Using K =", optimal_k, "for clustering.")

# Cluster the embeddings with K-Means using the chosen number of clusters
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans_final.fit_predict(embeddings)
df['cluster_label'] = cluster_labels

print("Clustering complete. Sample cluster labels:")
print(df[['translated_text', 'cluster_label']].head())

# Save the final clustered data to a new CSV file
df.to_csv('../data/processed/clustered_data_example.csv', index=False)
print("Clustering complete.")