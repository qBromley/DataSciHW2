import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

xValsTrain = pd.read_csv('p1/x1.csv')

yValsTrain = pd.read_csv('p1/y1.csv')

reg = linear_model.LinearRegression()

# Create and fit the linear regression model
reg = linear_model.LinearRegression()
reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])

# Plot the data and the regression line
plt.scatter([0, 1, 2], [0, 1, 2], color='blue', label='Data points')
plt.plot([0, 1, 2], reg.predict([[0, 0], [1, 1], [2, 2]]), color='red', label='Regression line')

# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.legend()

# Save the figure as an image (e.g., PNG)
image_path = "linear_regression_plot.png"
plt.savefig(image_path)

