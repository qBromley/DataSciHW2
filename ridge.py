import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
def ridge()
# Load data from CSV files
x = pd.read_csv('p1/x1.csv').values  # Ensure this is a NumPy array
y = pd.read_csv('p1/y1.csv').values
x_test = pd.read_csv('p1/x2.csv').values
y_test = pd.read_csv('p1/y2.csv').values


# Define the regularization strength (alpha)
alpha = 1.0  

# Perform Ridge Regression
ridge_model = Ridge(alpha=alpha, fit_intercept=False)  
ridge_model.fit(x, y)

# Get the Ridge coefficients
beta = ridge_model.coef_  
print("Set of coefficients:", beta.flatten())

# Predict on test data
y_pred_test = ridge_model.predict(x_test)
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
plt.savefig('ridge.png',format = 'png')
