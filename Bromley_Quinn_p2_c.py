import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import numpy as np
X =  pd.read_csv('p2/x1.csv')
c =  pd.read_csv('p2/c1.csv').values.ravel()
X2 = pd.read_csv('p2/x2.csv')
c2 = pd.read_csv('p2/c2.csv').values.ravel()

# Standardizing the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X2_scaled = scaler.fit_transform(X2)

# apply LDA
print(len(set(c)) - 1)
lda = LDA(n_components=len(set(c)) - 1)  
X_lda = lda.fit_transform(X_scaled, c)
# eigenvectors for lda
lda_eigenvectors = lda.scalings_


# Convert LDA results to DataFrame and add class labels
lda_df = pd.DataFrame(X_lda, columns=[f'LD{i+1}' for i in range(X_lda.shape[1])])
lda_df['Class'] = c

X_lda_projection = np.dot(X_scaled, lda_eigenvectors)
X2_lda_projection = np.dot(X2_scaled,lda_eigenvectors)

# Convert the projected data to a DataFrame
X_lda_df = pd.DataFrame(X_lda_projection, columns=[f'LD{i+1}' for i in range(X_lda_projection.shape[1])])
X_lda_df['Class'] = c  

X2_lda_df = pd.DataFrame(X2_lda_projection,columns=[f'LD{i+1}' for i in range(X2_lda_projection.shape[1])])
X2_lda_df['Class'] = c2
# Plotting 
plt.figure(figsize=(10, 7))
for label in X_lda_df['Class'].unique():
    plt.scatter(
        X_lda_df.loc[X_lda_df['Class'] == label, 'LD1'],
        X_lda_df.loc[X_lda_df['Class'] == label, 'LD2'],
        label=f'Class {label}'
    )
plt.xlabel('Linear Discriminant ')
plt.ylabel('Linear Discriminant ')
plt.title('LDA Projection of Dataset')
plt.legend()
plt.savefig('LDA_Training.png')
plt.clf()

plt.figure(figsize=(10, 7))
for label in X_lda_df['Class'].unique():
    plt.scatter(
        X2_lda_df.loc[X2_lda_df['Class'] == label, 'LD1'],
        X2_lda_df.loc[X2_lda_df['Class'] == label, 'LD2'],
        label=f'Class {label}'
    )
plt.xlabel('Linear Discriminant 1 (LD1)')
plt.ylabel('Linear Discriminant 2 (LD2)')
plt.title('LDA Projection of Validation Dataset')
plt.legend()
plt.savefig('LDA_Validation.png')

