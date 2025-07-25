# MSE [Mean Squared Error]
# Accuracy 

import numpy as np 



#----------------------MSE (Mean Squared Error)-----------------------#

# Formula for MSE = 1/n (y - y*)^2

def mse(y_true, y_pred):

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred) # key difference between np.asarray and np.array, 
                                                            # np.asarray does not make a copy of the array if it's already an np array 
                                                            # but np.array always makes a new np arra 
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true {y_true.shape} and y_pred {y_pred.shape} should have the same shape")
    return np.mean((y_true-y_pred) ** 2) # np.mean - name thing as 1/n (Divide by the number of samples 

#----------------------Accuracy-----------------------#

def accuracy(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape: # Checks weather both the arrays have the same shape
        raise ValueError(f"y_true {y_true.shape} and y_pred {y_pred.shape} should have the same shape")
    return np.mean(y_true == y_pred) # y_true == y_pred this lines checks every element in the y_true and y_pred array to provide us with a average score


#----------------------R^2 Score-----------------------#

# This is an statistical measure where we represent the relationship between the dependent and the independent variables 
# On the scale of 0-1. Defines how good the model actually is. If 1 that means it's perfect if 0 it means it's messed up 
# IF it's negative that means it exceeds the baseline error (yi - yi(bar)^2) Means it perfoms worse than it should 
# Numerator = SSres, Dinomenator = SStot 
# Subtracting 1 to get the desired result between 1 - 0

def r2(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("The shape should be equal")

    SSres = sum((y_true - y_pred)**2)
    mean_of_y_true = np.mean(y_true)
    SStot = sum((y_true - mean_of_y_true)**2)

    print(f"SStot: {SStot}, SSres: {SSres} ")

    return 1 - (SSres / SStot)

y_true = np.array([1,0,1,0]) 
y_pred = np.array([1,0,1,0])

print(r2(y_true, y_pred)) # Test 