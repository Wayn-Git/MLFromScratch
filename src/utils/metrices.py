# MSE [Mean Squared Error]
# Accuracy 

import numpy as np 

# Formula for MSE = 1/n (y - y*)^2

def MSE(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred) # key difference between np.asarray and np.array, 
                                                            # np.asarray does not make a copy of the array if it's already an np array 
                                                            # but np.array always makes a new np array
    return np.mean((y_true-y_pred) ** 2) # np.mean - name thing as 1/n (Divide by the number of samples n)