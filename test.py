import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV files
x = pd.read_csv('p1/x1.csv').values  # Ensure this is a NumPy array
y = pd.read_csv('p1/y1.csv').values
x_test = pd.read_csv('p1/x2.csv').values
y_test = pd.read_csv('p1/y2.csv').values


# Calculate the pseudo-inverse solution
# beta = (X^T X)^(-1) X^T y
beta = np.linalg.inv(x.T @ x) @ x.T @ y
print("Set of coefficients:", beta.ravel())  # Flatten for a cleaner print format

# Predict on training data to calculate R^2 on training set
y_pred_train = x @ beta
rss_train = np.sum((y - y_pred_train) ** 2)
tss_train = np.sum((y - np.mean(y)) ** 2)
r2_train = 1 - rss_train / tss_train
print("R^2 on training data:", r2_train)

# Predict on test data
y_pred_test = x_test @ beta
rss_test = np.sum((y_test - y_pred_test) ** 2)
tss_test = np.sum((y_test - np.mean(y_test)) ** 2)
r2_test = 1 - rss_test / tss_test
print("R^2 on test data:", r2_test)

# Plot ground truth vs predictions for test data
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Ground Truth (y_test)")
plt.ylabel("Predicted (y_pred_test)")
plt.title("Scatter Plot of Ground Truth vs Predictions (Test Data)")
plt.savefig('test.png',format = 'png')
