import numpy as np
from numba import cuda
import math

class LogisticRegressionGPU:
    def __init__(self, learning_rate=0.2, epsilon=2e-2, max_iter=1000):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.w = None

    def fit(self, X, y):
        # Add bias to the input features
        X = self.add_bias(X)
        X = X.astype(float)
        y = y.astype(float)

        # Set the size of the GPU thread blocks and the number of blocks needed
        block_size = 1024
        grid_size = (X.size + block_size - 1) // block_size

        # Initialize the weights of the logistic regression model
        self.w = np.zeros(X.shape[1], dtype=float).reshape(1, X.shape[1])
        self.w[0][0] = 1

        # Transfer the data y and w to the GPU device
        y_d = cuda.to_device(y)
        w_d = cuda.to_device(self.w.copy())

        grad_loss_d = cuda.to_device(self.w.copy())

        for _ in range(self.max_iter):
            # Forward pass
            X_d1 = cuda.to_device(X.copy())
            res = np.zeros(X.shape[0], dtype=float).reshape(1, X.shape[0])
            res_d = cuda.to_device(res)
            X2 = X_d1.T

            self.vector_matrix_mul[grid_size, block_size](w_d, X2)
            cuda.synchronize()

            self.matrix_col_sum[grid_size, block_size](X2, res_d)
            cuda.synchronize()

            self.sigmoid[grid_size, block_size](X_d1, res_d)
            cuda.synchronize()

            # Backward pass
            self.substract[grid_size, block_size](y_d, res_d)
            cuda.synchronize()

            X3 = cuda.to_device(X.copy())
            self.vector_matrix_mul[grid_size, block_size](res_d, X3)
            cuda.synchronize()

            self.matrix_col_sum[grid_size, block_size](X3, grad_loss_d)
            cuda.synchronize()

            # Update the weights using the gradient and the learning rate
            grad_loss = grad_loss_d.copy_to_host() * (self.learning_rate * 1. / X.shape[0])
            grad_loss_d = cuda.to_device(grad_loss)

            self.substract[grid_size, block_size](grad_loss_d, w_d)
            cuda.synchronize()

            # Calculate the magnitude of the gradient vector using the L2 norm
            grad_norm = np.array([[0]], dtype=float)
            grad_norm_d = cuda.to_device(grad_norm)
            self.norm2[grid_size, block_size](grad_loss_d, grad_norm_d)
            cuda.synchronize()

            # Check whether the gradient magnitude is smaller than the epsilon threshold
            grad_norm = grad_norm_d.copy_to_host()

            if math.sqrt(grad_norm[0][0]) <= self.epsilon or np.linalg.norm(self.w - w_d.copy_to_host()) <= self.epsilon:
                break

        self.w = w_d.copy_to_host()

    def predict(self, X):
        """
        Predicts the labels for a given set of data using the trained logistic regression weights.

        Args:
        X: numpy array, shape (m, n), input data

        Returns:
        predictions: numpy array, shape (m,), predicted labels
        """
        # Add bias to x
        X = self.add_bias(X)

        # These lines set the size of the GPU thread blocks and the number of blocks needed based on the size of the input data X
        block_size = 200
        grid_size = (X.size + block_size - 1) // block_size

        X_d = cuda.to_device(X)
        w_d = cuda.to_device(self.w)
        res = np.zeros(X.shape[0], dtype=float).reshape(1, X.shape[0])
        res_d = cuda.to_device(res)
        X2 = X_d.T

        self.vector_matrix_mul[grid_size, block_size](w_d, X2)
        cuda.synchronize()

        self.matrix_col_sum[grid_size, block_size](X2, res_d)
        cuda.synchronize()

        self.sigmoid[grid_size, block_size](X_d, res_d)
        cuda.synchronize()

        # Round the predictions to the nearest integer (0 or 1)
        predictions = np.round(res_d.copy_to_host())

        return predictions

    @staticmethod
    @cuda.jit
    def vector_matrix_mul(v, m):
        # Get thread index
        start = cuda.grid(1)
        stripe = cuda.gridsize(1)
        # Check if thread index is within size of the vector
        for i in range(start, v.shape[1], stripe):
            # perform the dot product
            for j in range(m.shape[1]):
                m[i][j] = v[0][i] * m[i][j]

    @staticmethod
    @cuda.jit
    def matrix_col_sum(m, result):
        # Get thread index
        start = cuda.grid(1)
        stripe = cuda.gridsize(1)
        # Check if thread index is within size of the vector
        for j in range(start, m.shape[1], stripe):
            # perform the dot product
            result[0][j] = 0
            for i in range(m.shape[0]):
                result[0][j] += m[i][j]

    @staticmethod
    @cuda.jit
    def sigmoid(X, res):
        """
        w is a vector of shape (1, n)
        X is a matrix of shape (m, n)
        m is the number of examples
        n is the number of features
        """
        # Get thread index
        start = cuda.grid(1)
        stripe = cuda.gridsize(1)
        # Check if thread index is within size of the vector
        for i in range(start, X.shape[0], stripe):
              # perform the dot product
              res[0][i] = 1 / (1 + math.exp(-res[0][i]))

    @staticmethod
    @cuda.jit
    def substract(vec1, res2):
        """
        this function makes an element-wise subtraction, i.e., res2 - vec1

        vec1 and res2 are vectors of shape (1, m)
        m is the number of examples

        the result of the subtraction is stored in res2
        """
        # Get thread index
        start = cuda.grid(1)
        stripe = cuda.gridsize(1)
        # Check if thread index is within size of the vector
        for i in range(start, vec1.shape[1], stripe):
            cuda.atomic.add(res2[0], i, -vec1[0][i])

    @staticmethod
    @cuda.jit
    def norm2(vec, res):
        """
        vec is a vector of shape (1, n)
        """
        # Get thread index
        start = cuda.grid(1)
        stripe = cuda.gridsize(1)
        # Check if thread index is within size of the vector
        for i in range(start, vec.shape[1], stripe):
            # perform the dot product
            cuda.atomic.add(res[0], 0, vec[0][i] * vec[0][i])

    @staticmethod
    def add_bias(X):
        return np.hstack((np.ones((X.shape[0], 1)), X))

