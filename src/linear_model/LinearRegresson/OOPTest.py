import numpy as np

class LinearRegression:
    
    def __init__(self, epochs=1000, learning_rate=1e-2):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.w = None
        self.b = None
    def fit(self, x, y):
        self.w = np.random.rand()
        self.b = 0.0
        
        for i in range(self.epochs):
            
            y_pred = (self.w * x) + self.b
            
            loss = np.mean(y_pred - y) ** 2
            
            w_gradient = 2 * np.mean((y_pred - y) * x)
            b_gradient = 2 * np.mean(y_pred - y)
            
            self.w -= w_gradient * self.learning_rate
            self.b -= b_gradient * self.learning_rate
            
            if i % 100 == 0:
                print(f"Epochs: {i} w: {self.w:.4f} b: {self.b:.4f} Loss: {loss:.4f}")
                
    def predict(self, x):
        return (self.w * x) + self.b
                
                
x = np.array([1,2,3,4,5])
y = np.array([2,4,5,6,7])

model = LinearRegression()
model.fit(x,y)
prediction = model.predict(np.array([8,9,10]))
print(prediction)
            