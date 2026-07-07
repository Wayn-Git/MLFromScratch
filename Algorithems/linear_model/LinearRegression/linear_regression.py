# Linear Regression 

# This is where we find the best fit for our trueValue point
# Weights: This is a parameter that adjusts how much influence should the value of first neuron have on the proceeding neuron
# Bias: A systematic error that causes due to the wrong assumption of the model 
# The difference between the true and the predicted value 
# (We start with a random set of weights and baises that are then tweaked with the help of gradient decent  (An Optimization algorithem))

import numpy as np

class LinearRegression:
    
    def __init__(self, epochs=1000, learning_rate=1e-2): # Here x is the feature and y is the target while Epochs refer is each time we go through the data set and tune the weight and bias
                                                        # Learning rate affects how fast or slow the model learns, If too high it takes bigger steps missing the best data point, if too low it may take too long to learn and possibly get stuck. Best to find the best balance 
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.w = None # We start with a random weight and initially adjust it with gradient descent (Optimization algorithem). 
                         # This determines how strong a input feature affects the target
        self.b = None # Similarly we start with the origin. The bias basically contorls the position of the line.
    def fit(self, x, y):
        self.w = np.random.rand()
        self.b = 0.0
        
        for i in range(self.epochs):
            
            y_pred = (self.w * x) + self.b #Linear Regression Formula Y = Mx + b. here M = w, x = x, b = b. W contrls how much influence each feature has on the target value while the b helps to shift the position of the line
            
            loss = np.mean(y_pred - y) ** 2 # Loss function to calculate the error MSE Mean Squared Error
            
            w_gradient = 2 * np.mean((y_pred - y) * x)  # Gradient descent an optimzation algorithem used to move the values of weight and biases to get a perfect value for the prediction to be accurate. We multiply x because it helps the model to learn and tells the model how each feature contribute and how each affect the accuracy
            b_gradient = 2 * np.mean(y_pred - y)
            
            self.w -= w_gradient * self.learning_rate # Moving in the direction where the loss is the least is why we use -= 
            self.b -= b_gradient * self.learning_rate
            
            if i % 100 == 0:
                print(f"Epochs: {i} w: {self.w:.4f} b: {self.b:.4f} Loss: {loss:.4f}")
                
    def predict(self, x):
        return (self.w * x) + self.b
                
                
x = np.array([1,2,3,4,5])
y = np.array([3,4,5,6,7])



model = LinearRegression()
model.fit(x,y)
prediction = model.predict(np.array([8,9,10]))
print(prediction)
            