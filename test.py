import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import numpy as np

# Load training data
X = pd.read_csv('p2/x1.csv',header = None)
c = pd.read_csv('p2/c1.csv',header = None).values.ravel()

# Load validation data
X_val = pd.read_csv('p2/x2.csv',header = None)  # Replace with the path to your validation dataset
c_val = pd.read_csv('p2/c2.csv',header = None)
# Standardize the training data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply LDA to training data
lda = LDA(n_components=len(set(c)) - 1)
X_lda = lda.fit_transform(X_scaled, c)

# Get the transformation matrix (eigenvectors)
lda_eigenvectors = lda.scalings_

# Project the training data onto the LDA space
X_lda_projection = np.dot(X_scaled, lda_eigenvectors)

# Standardize the validation data using the same scaler
X_val_scaled = scaler.transform(X_val)

# Project the validation data onto the LDA space using the eigenvectors
X_val_lda_projection = np.dot(X_val_scaled, lda_eigenvectors)

# Convert projections to DataFrames for plotting
X_lda_df = pd.DataFrame(X_lda_projection, columns=[f'LD{i+1}' for i in range(X_lda_projection.shape[1])])
X_lda_df['Class'] = c

X_val_lda_df = pd.DataFrame(X_val_lda_projection, columns=[f'LD{i+1}' for i in range(X_val_lda_projection.shape[1])])
X_val_lda_df['Class'] = c_val
# Plotting the training data
plt.figure(figsize=(10, 7))
for label in X_lda_df['Class'].unique():
    plt.scatter(
        X_lda_df.loc[X_lda_df['Class'] == label, 'LD1'],
        X_lda_df.loc[X_lda_df['Class'] == label, 'LD2'],
        label=f'Class {label}'
    )
plt.xlabel('Linear Discriminant 1 (LD1)')
plt.ylabel('Linear Discriminant 2 (LD2)')
plt.title('LDA Projection of Training Dataset')
plt.legend()
plt.savefig('LDA_Training.png')
plt.clf()  # Clear the plot

# Plotting the validation data (without class labels)
plt.figure(figsize=(10, 7))
for label in X_lda_df['Class'].unique():
    plt.scatter(
        X_val_lda_df.loc[X_val_lda_df['Class'] == label, 'LD1'],
        X_val_lda_df.loc[X_val_lda_df['Class'] == label, 'LD2'],
        label=f'Class {label}'
    )
plt.xlabel('Linear Discriminant 1 (LD1)')
plt.ylabel('Linear Discriminant 2 (LD2)')
plt.title('LDA Projection of Validation Dataset')
plt.legend()
plt.savefig('LDA_Validation.png')
