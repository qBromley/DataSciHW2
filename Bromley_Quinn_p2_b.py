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
pca = PCA()

pca.fit(X_scaled)
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance')
plt.savefig('scree.png')
