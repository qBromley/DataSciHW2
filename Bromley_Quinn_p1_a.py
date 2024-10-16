from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt
reg = linear_model.LinearRegression()
# Reading in data
x = pd.read_csv('p1/x1.csv')
y = pd.read_csv('p1/y1.csv')
x_test = pd.read_csv('p1/x2.csv')
y_test = pd.read_csv('p1/y2.csv')
#calculating OLS 
reg.fit(x,y)
#OlS coefficent matrix
ans = reg.coef_
print('set of coeff: ',ans)
print('R^2: ', reg.score(x,y))

x_test_aligned = x_test.reindex(columns=x.columns, fill_value=0)
y_pred = reg.predict(x_test_aligned)
print("R^2 predict :", reg.score(x_test_aligned,y_pred))

y_test_flat = y_test.values.ravel()  # Flatten y_test to 1D if it's not
y_pred_flat = y_pred.ravel()  # Flatten y_pred to 1D if it's not

# Create DataFrame for comparison
comparison_df = pd.DataFrame({'Actual': y_test_flat, 'Predicted': y_pred_flat})
plt.figure(figsize=(8, 6))
plt.scatter(y, comparison_df['Predicted'], alpha=0.6)

plt.xlabel("Ground Truth (y_test)")
plt.ylabel("Predictions (y_pred)")
plt.title("Scatter Plot of Ground Truth vs. Predictions")
plt.savefig('comparision.png',format = 'png' )

