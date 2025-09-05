from Algorithems.linear_model.LinearRegression import linear_regression
from Algorithems.linear_model.LogisticRegression import logistic_regression
from Algorithems.UtilMetricies import metrices
import numpy as np
import math

# x = np.array([1,2,3,4,5,6,7])
# y = np.array([2*x for x in x])

# y_pred = linear_regression.linear_regression(x,y)

x = np.array([1, 2, 3, 4, 5])
y = np.array([0, 0, 1, 1, 1])

# y_pred_prob = logistic_regression.LogisticRegression(x, y)
# y_pred = (y_pred_prob >= 0.5).astype(int)

# print("Accuracy:", metrices.accuracy(y, y_pred))
# print("MSE:", metrices.mse(y, y_pred))
# print("RS2:", metrices.r2(y, y_pred))