import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Create Sample Customer Purchase Data (Manual Data)
data = {
    'customer_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'total_spent': [500, 1500, 700, 8000, 200, 300, 10000, 9000, 400, 6000],  # Total money spent
    'num_purchases': [5, 15, 7, 80, 2, 3, 100, 95, 4, 60],  # Number of purchases made
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Step 2: Select features for clustering (excluding customer_id)
X = df[['total_spent', 'num_purchases']]

# Step 3: Normalize the data (K-means works better with scaled data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply K-means clustering
k = 3  # Number of clusters (you can experiment with different values)
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Step 5: Visualize the clusters
plt.figure(figsize=(8, 6))
for cluster in range(k):
    clustered_data = df[df['cluster'] == cluster]
    plt.scatter(clustered_data['total_spent'], clustered_data['num_purchases'], label=f'Cluster {cluster}')

# Mark centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='red', label='Centroids')

plt.xlabel('Total Spent ($)')
plt.ylabel('Number of Purchases')
plt.title('Customer Segmentation Using K-Means')
plt.legend()
plt.show()

# Step 6: Print Clustered Data
print(df)
