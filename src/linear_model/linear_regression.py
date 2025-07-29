# Linear Regression 

# This is where we find the best fit for our trueValue point
# Weights: This a parameter parameter that adjusts how much influence should the value of first neuron have on the proceeding neuron
# Bias: A systematic error that causes due to the wrong assumption of the model 
# The difference between the true and the predicted value 
# (We start with a random set of weights and baises that are then tweaked with the help of gradient decent  (An Optimization algorithem))



import numpy as np


def linear_regression(x, y, epochs=1000, lr_rate=1e-2): # Here x is the feature and y is the target while Epochs refer is each time we go through the data set and fine tune the weight and bias
                                                        # Learning rate affects how fast or slow the model learns, If too high it takes bigger steps missing the best data point, if too low it may take too long to learn and possibly get stuck. Best to find the best balance 
    w = np.random.rand() # We start with a random weight and initially adjust it with gradient descent (Optimization algorithem). This determines how strong a input feature affects the target
    b = 0.0 # Similarly we start with the origin. The bias basically contorles the position of the line (if I'm not wrong). 

    for i in range(epochs): 
        y_pred = np.dot(w, x) + b #Linear Regression Formula Y = Mx + b. here M = w, x = x, b = b. W contrls how much influence each feature has on the target value while the b helps to shift the position of the line
        loss = np.mean((y_pred - y) **2) # Loss function to calculate the error MSE Mean Squared Error

        w_gradient = 2 * np.mean((y_pred - y) * x) # Gradient descent an optimzation algorithem used to move the values of weight and biases to get a perfect value for the prediction to be accurate. We multiply x because it helps the model to learn and tells the model how each feature contribute and how each affect the accuracy

        b_gradient = 2 * np.mean((y_pred - y))

        w -= w_gradient * lr_rate # Moving in the direction where the loss is the least is why we use -= 
        b -= b_gradient * lr_rate

        if i % 100 == 0:
            print(f"Epoch {i}: Prediction: {y_pred:}, Loss: {loss:.2f}")
        


x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])
linear_regression(x, y)





