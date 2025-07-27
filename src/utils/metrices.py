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

#----------------------RMSE Score-----------------------#

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

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
# Numerator = SSres (sum of the squares of residuals ), Dinomenator = SStot (Total Sum of Squares) 
# Subtracting 1 to get the desired result between 1 - 0

def r2(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("The shape should be equal")

    SSres = sum((y_true - y_pred)**2)
    mean_of_y_true = np.mean(y_true)
    SStot = sum((y_true - mean_of_y_true)**2)

    return 1 - (SSres / SStot)



#----------------------Precision-----------------------#

# Precision is mainly used in Classification problems
# The problems where we deal with "yes" or "no"/ "0" or "1"

# Example: We provide a image of a dog to a model and we ask it weather it's a dog or cat (dog can be 1 and cat can be 0)
# The model would respond with either 1 or 0 based on the image
# Precision calculates how many 1 is actually 1. This helps us to reduce the false 1

# Precision is a matric where we calculate how far off our model is actually from the true actual thing 
# Include sjust the positive aspects. In simple words how many "yes" are actually "yes"

# Formula for Precision: True Positive (TP) (1) / (True Positive (TP) (1) - False Positive (FP) (Was supposed to be 1 but it's a 0) )

def precision(y_true, y_pred):

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    if(y_true.shape != y_pred.shape):
        raise ValueError(f"y_true: {y_true.shape} and y_pred {y_pred.shape} must haev the same shape")
    
    y_true_bool, y_pred_bool = y_true.astype(bool), y_pred.astype(bool)

    true_positive = np.sum(y_true_bool & y_pred_bool)
    false_positive = np.sum(~y_true_bool & y_pred_bool)

    numerator = true_positive
    denominator = true_positive + false_positive 

    precision = numerator / denominator

    if(denominator == 0):
        return 0
    else: 
        return precision
    

#----------------------Recall-----------------------#

# Recall is a matric where we calculate how many actual values did the model predict to be false 
# The Dog and Cat image: 
# Lets say the image was actually a dog but the model predicted it to be Cat
# This is where Recall comes in action. It calculates how many True Values were Predicted as false


# Formula for Recall: True Positive(1) / True Positive(1) - False Negative (0)


def recall(y_true, y_pred):

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    if(y_true.shape != y_pred.shape):
        raise ValueError(f"y_true: {y_true.shape} and y_pred {y_pred.shape} must haev the same shape")
    
    y_true_bool, y_pred_bool = y_true.astype(bool), y_pred.astype(bool)

    true_positive = np.sum(y_true_bool & y_pred_bool)
    false_negative = np.sum(y_true_bool & ~y_pred_bool)

    numerator = true_positive
    denominator = true_positive + false_negative

    recall = numerator / denominator

    if(denominator == 0):
        return 0
    else: 
        recall



#----------------------F1 Score-----------------------#

# Combines the recall and precision into a single value that ranges from 0 to 1

def f1_score(y_true, y_pred):

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)    

    if(y_true.shape != y_pred.shape):
        raise ValueError(f"y_true: {y_true.shape} and y_pred {y_pred.shape} must haev the same shape")

    y_true_bool, y_pred_bool = y_true.astype(bool), y_pred.astype(bool)

    f1_score = 2 * (
     precision(y_true, y_pred) * recall(y_true, y_pred)
     / (precision(y_true, y_pred) + recall(y_true, y_pred))
    )

    
    return f1_score

    
