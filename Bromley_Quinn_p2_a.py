import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
X =  pd.read_csv('p2/x1.csv')
c =  pd.read_csv('p2/c1.csv')

# Standardizing the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA to reduce to two components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['nvert PCA results to DataFrame and add class labelsClass'] = c

# Plotting the results
plt.figure(figsize=(10, 7))
for label in pca_df['Class'].unique():
    plt.scatter(
        pca_df.loc[pca_df['Class'] == label, 'PC1'],
        pca_df.loc[pca_df['Class'] == label, 'PC2'],
        label=f'Class {label}'
    )
plt.xlabel('Principal Component')
plt.ylabel('Principal Component')
plt.title('PCA of Dataset')
plt.legend()
plt.savefig('pca.png')
plt.close()


