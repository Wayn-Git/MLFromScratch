import numpy as np 

def sigmoid(z): 
    return 1/(1+np.exp(-z))


def LogisticReg(x, y , epochs = 1000, lr_rate = 1e-2):
  w = 0  
  b = 0.0 

  for i in range(epochs):
   linear_model = w * x + b
   y_pred = sigmoid(linear_model)

   loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

   w_gradient = np.mean((y_pred - y) * x)
   b_gradient = np.mean(y_pred - y)


   w -= w_gradient * lr_rate
   b -= b_gradient * lr_rate

   

   result = sigmoid(y_pred)

   if i % 100 == 0:
     print(f"Epoch {i}: loss = {loss}, predictions = {y_pred}")


x = np.array([1, 2, 3, 4, 5])
y = np.array([0, 0, 1, 1, 1])
LogisticReg(x, y)

 
  
  
  
