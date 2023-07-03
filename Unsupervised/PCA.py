from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
df = pd.DataFrame(pca.fit_transform(df), columns = ['PC1', 'PC2'])
df
