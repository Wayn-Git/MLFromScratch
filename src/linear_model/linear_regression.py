# Linear Regression 

# This is where we find the best fit for our trueValue point
# Weights: This a parameter parameter that adjusts how much influence should the value of first neuron have on the proceeding neuron
# Bias: A systematic error that causes due to the wrong assumption of the model 
# The difference between the true and the predicted value 
# (We start with a random set of weights and baises that are then tweaked with the help of gradient decent  (An Optimization algorithem))



import numpy as np




def linear_regression(trueValue, predValue):
    trueValue = trueValue/np.max(trueValue)

    w = np.random.rand(trueValue.shape)

    b = 0

    lr_rate = 1e-2

    for i in range(1000):

        pred_value = np.dot(trueValue, w) + b

        loss_function = (predValue - trueValue) ** 2

        w_gradient =  2 * (predValue - trueValue) * trueValue

        b_gradient = 2*(predValue - trueValue)

        w -= w_gradient * lr_rate
        b -= b_gradient * lr_rate

        if i % 100 == 0: 
            print(predValue)
    




