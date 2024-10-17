import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
def ridge(x, y, xT, yT, a):
    ridge_model = Ridge(alpha=a, fit_intercept=False)  
    ridge_model.fit(x, y)
    y_pred = ridge_model.predict(xT)
    
    # Calculate RSS 
    rss_test = np.sum((yT - y_pred) ** 2)
    
    # Calculate TSS 
    tss_test = np.sum((yT - np.mean(yT)) ** 2)
    
    # Calculate R^2
    r2_test = 1 - rss_test / tss_test
    return r2_test
    
    
# Load data from CSV files
x = pd.read_csv('p1/x1.csv').values  # Ensure this is a NumPy array
y = pd.read_csv('p1/y1.csv').values
x_test = pd.read_csv('p1/x2.csv').values
y_test = pd.read_csv('p1/y2.csv').values


alpha_values = np.logspace(-3, 3, 100)
R2List = []
for alpha in alpha_values:
    r2_value = ridge(x, y, x_test, y_test, alpha)
    R2List.append(round(r2_value, 4))  #

print("Alpha values:", alpha_values)
print("R^2 values:", R2List)

plt.figure(figsize=(10, 6))
plt.plot(alpha_values, R2List, marker='o', linestyle='-', color='b')
plt.xscale('log')  
plt.xlabel("Alpha (Regularization Strength)")
plt.ylabel("R^2 Score on Test Data")
plt.title("R^2 Score vs Alpha in Ridge Regression")
plt.savefig('ridge.png')