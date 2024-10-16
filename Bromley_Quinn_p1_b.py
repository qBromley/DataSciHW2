import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
n = 2000     # Number of examples
d = 5       # Number of predictors
dd = 400     # Number of noisy channels
std = 0.2   # Standard deviation of additive noise

# Generate predictors
x = np.random.randn(n, d)            # Normally distributed predictors (IVs)
x = np.hstack([np.ones((n, 1)), x])  # Add intercept
w = np.random.randn(d + 1, 1)        # Forward model (from x to y)
w /= np.sqrt(np.sum(w**2))           # Normalize regression coeffs (unit length)
y = x @ w                            # Generate dependent variable (DV)

# Add noise to DV
y += std * np.random.randn(*y.shape)

# Add noisy channels to predictors
x = np.hstack([x, np.random.randn(n, dd)])

# Shuffle predictors and noise channels
ix = np.random.permutation(x.shape[1])
x = x[:, ix]

# Split data into training and test sets
x1 = x[:n // 2, :]  # Training data
y1 = y[:n // 2, :]
x2 = x[n // 2:, :]  # Test data
y2 = y[n // 2:, :]



# Concatenate training and test data for shuffling
x = np.vstack([x1, x2])
y = np.vstack([y1, y2])

# Initialize list to store R^2 values
r2_scores = []

# Perform 200 repetitions
for rep in range(200):
    # Shuffle data
    ix = np.random.permutation(x.shape[0])
    x = x[ix, :]
    y = y[ix, :]

    # Re-split data into training and test sets
    x1 = x[:n // 2, :]
    y1 = y[:n // 2, :]
    x2 = x[n // 2:, :]
    y2 = y[n // 2:, :]

    # Perform least squares regression
    w_pred = np.linalg.inv(x1.T @ x1) @ x1.T @ y1  # Pseudo-inverse solution

    # Predict on training and test data
    y1_pred = x1 @ w_pred
    y2_pred = x2 @ w_pred

    # Calculate R^2 for test data
    rss = np.sum((y2 - y2_pred) ** 2)
    tss = np.sum((y2 - np.mean(y2)) ** 2)
    r2 = 1 - rss / tss
    r2_scores.append(r2)

    # Plot ground truth vs predicted values
    plt.plot(y1, y1_pred, 'b.', alpha=0.5, label="Training" if rep == 0 else "")
    plt.plot(y2, y2_pred, 'r.', alpha=0.5, label="Test" if rep == 0 else "")

# Show R^2 plot
plt.xlabel("Ground Truth")
plt.ylabel("Predicted")
plt.legend()
plt.title("Ground Truth vs Predicted (Training in Blue, Test in Red)")
plt.savefig('GrdTruthVsPredict.png',format = 'png' )
plt.close()

# Calculate and print mean R^2
print("Mean R^2:", np.mean(r2_scores))

# Plot histogram of R^2 scores
plt.figure()
plt.hist(r2_scores, bins=25, color="c", edgecolor="black")
plt.xlabel("R^2 Score")
plt.ylabel("Frequency")
plt.title("Distribution of R^2 Scores")
plt.savefig('FreqVsR2.png',format = 'png' )
plt.close()
# 