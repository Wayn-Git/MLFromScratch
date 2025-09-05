from Algorithems.linear_model.LinearRegression import linear_regression
from Algorithems.UtilMetricies import metrices
import numpy as np
import math

x = np.array([1,2,3,4,5,6,7])
y = np.array([2*x for x in x])

y_pred = linear_regression.linear_regression(x,y)
print("Accuracy:", metrices.linear_accuracy(y, y_pred))
print("MSE:", metrices.mse(y, y_pred))
print("RS2:", metrices.r2(y, y_pred))