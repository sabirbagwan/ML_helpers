import seaborn as sns
sns.set_style("darkgrid")

max_clusters = 50
inertia_values = []

k_values = range(1, max_clusters)
for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(df)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
sns.lineplot(x=k_values, y=inertia_values, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for K-means Clustering')
plt.xticks(range(0, max_clusters, 10))
plt.show()
