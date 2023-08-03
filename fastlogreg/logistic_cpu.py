import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.2, epsilon=2e-2):
        self.lr = lr
        self.epsilon = epsilon
        self.w = None

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def hs(x, w):
        return np.dot(w, x.T)

    def fit(self, X, Y):
        # add bias to x
        X = self.add_bias(X)
        self.w = np.zeros(X.shape[1]).reshape(1, X.shape[1])  # initialize weights vector

        while True:
            gradient = ((self.sigmoid(self.hs(X, self.w)) - Y) @ X) * (1 / X.shape[0])
            self.w -= self.lr * gradient  # gradient descent
            
            if np.linalg.norm(gradient) <= self.epsilon:
                return self.w

    def predict(self, X):
        # add bias to x
        X = self.add_bias(X)
        # Calculate the dot product of X and the weight vector w
        z = np.dot(X, self.w.T)

        # Calculate the sigmoid of the dot product
        predictions = 1 / (1 + np.exp(-z))

        # Round the predictions to the nearest integer (0 or 1)
        predictions = np.round(predictions)

        return predictions

    @staticmethod
    def add_bias(X):
        return np.hstack((np.ones((X.shape[0], 1)), X))
