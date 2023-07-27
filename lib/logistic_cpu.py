import numpy as np

def sigmoid(x):
    """
    description: activation function (sigmoid)
    args:
        x: scalar
    return: sigmoid of x
    """
    return 1 / (1 + np.exp(-x))

def hs(x,w):
    """
    description: logistic regression hypothesis
    args:
        x: vector of features
        w: vector of weights
    return: matrcial product of x and w.T
    """
    return np.dot(w,x.T)

def LogisticRegression(X,Y, lr = 7, epsilon = 1e-4):

    # add bais to x
    X = add_bias(X)

    w = np.zeros(X.shape[1]).reshape(1,X.shape[1]) # initialize weights vector

    while True:

        gradient = (( sigmoid(hs(X,w))-Y)  @  X)  *  (1/X.shape[0])

        w -= lr* gradient # gradient descent
        
        if np.linalg.norm(gradient) <=epsilon:
                    return w

def predict(X, w):
    """
    Predicts the labels for a given set of data using the trained logistic regression weights.

    Args:
    X: numpy array, shape (m, n), input data
    w: numpy array, shape (1, n), learned logistic regression weights

    Returns:
    predictions: numpy array, shape (m,), predicted labels
    """
    # add bais to x
    X = add_bias(X)
    # Calculate the dot product of X and the weight vector w
    z = np.dot(X, w.T)

    # Calculate the sigmoid of the dot product
    predictions = 1 / (1 + np.exp(-z))

    # Round the predictions to the nearest integer (0 or 1)
    predictions = np.round(predictions)

    return predictions
