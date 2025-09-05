# Logistic Regression is a classification algorithem that transforms the linear output of a linear model into a probability 
# it takes input features (which can be numerical or transformed from other data types, like text),
# computes a linear combination of these features

# Using the sigmoid function (Formula: 1/1+e^-x) The sigmoid function takes in a real world value and converts it into


# A probability that comes between the range of 1 and 0 
# Threshold: Probability > 0.5 -> Class = 1 else 0 

#---------Example--------#

# In spam detection, logistic regression converts email text into numerical features. 
# It learns the relationship between these features and whether emails are spam or not,
#  outputting probabilities so you can classify new emails as "spam" or "not spam".


import numpy as np 

def sigmoid(z): 
    return 1/(1+np.exp(-z))


def LogisticRegression(x, y , epochs = 1000, lr_rate = 1e-2):
    w = 0   # Doesn't really matter if it's random or 0. Will be optimized later
    b = 0.0 

    for i in range(epochs):
       
        #------Linear Model---------
        linear_model = w * x + b

        #------Prediction and Convertion---------
        y_pred = sigmoid(linear_model)

        # Clip to avoid log(0)
        eps = 1e-15   # Epsilon (eps) in np.clip is just a tiny constant to prevent mathematical errors when taking the log of probabilities.
        y_pred = np.clip(y_pred, eps, 1 - eps)

        #------Loss Function---------
        """
        Binary (0,1) Cross Entropy Loss 
  
        Formula: -1/n[(yi * log(pi)  + (1-yi) log(1-pi)]

        Here: 1/n refers to the mean 
              yi refers to the true value
              pi refers to the predicted value 

        How It Works:
        If the true label is 1 and the model predicts a probability close to 1 the loss for that sample is very small
        If the model predicts a probability close to 0 (wrong) when the true label is 1 the loss becomes very large
        The same logic applies for true label 0
        The loss penalizes confident wrong predictions much more than less confident ones
        
        Why Logarithms: 
        Logarithms strongly punishes the probabilities that are too far from the actual label which encourages the model to produce a 
        confident and accurate label
        """
        loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)) 

        #------Gradients---------# (Gradients are optimzing algorithem that are mainly used to adjust the value of the weights and bias with respect average of the predicted value, actual value and faetures)
        w_gradient = np.mean((y_pred - y) * x)
        b_gradient = np.mean(y_pred - y)

        w -= w_gradient * lr_rate # Adjusting the weight according to the gradient and the learning rate 
        #                         (The learning rate controls how fast we change the weights/Bias) Too high can make it skip the perfect weight and bais and too low can make it stuck so we have to find the sweet spot
        b -= b_gradient * lr_rate

        if i % 100 == 0:
            print(f"Epoch {i}: loss = {loss}, predictions = {y_pred}")
    linear_model_return = np.dot(w, x) + b
    y_pred = sigmoid(linear_model_return)
    return y_pred


x = np.array([1, 2, 3, 4, 5])
y = np.array([0, 0, 1, 1, 1])
LogisticRegression(x, y)
