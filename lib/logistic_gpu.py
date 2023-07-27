# Import of necessary libraries
from numba import cuda
import math as math

# Vector-Matrix Multiplication Kernel
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

@cuda.jit
def matrix_col_sum(m, result):
    # Get thread index
    start = cuda.grid(1)
    stripe = cuda.gridsize(1)
    # Check if thread index is within size of the vector
    for j in range(start, m.shape[1], stripe):
        # perform the dot product
        result[0][j]=0
        for i in range(m.shape[0]):
             result[0][j]+= m[i][j]

# Sigmoid Function
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
          res[0][i]= 1/(1+math.exp(-res[0][i]))

# Element-wise vector subtraction function using CUDA
@cuda.jit
def substract(vec1, res2):
  """
  this fuction make an element wise substruction  i.e.  res2-vec1

  vec1 and res2 are vectors of shape 1 m
  m is the number of examples

  the result of the substraction is stored in res2
  """
  # Get thread index
  start = cuda.grid(1)
  stripe = cuda.gridsize(1)
  # Check if thread index is within size of the vector
  for i in range(start, vec1.shape[1], stripe):
    cuda.atomic.add(res2[0],i,-vec1[0][i])

# CUDA function for computing L2 norm of a vector
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
        cuda.atomic.add(res[0],0,vec[0][i]*vec[0][i])

def add_bias(X):
    """
    Adds a column of 1's to the input feature matrix X, corresponding to the bias term.

    Args:
    - X (numpy.ndarray): input feature matrix of shape (n_samples, n_features)

    Returns:
    - X_new (numpy.ndarray): feature matrix with an additional column of 1's of shape (n_samples, n_features+1)
    """

    # Insert a column of 1's as the first column of X
    X_new = np.insert(X, 0, 1, axis=1)

    return X_new

# Logistic Regression on GPU using CUDA
def logistic_regression( X, y, learning_rate, epsilon=1e-4):
  """
     X (the input features),
     y (the labels),
     learning_rate (the step size used during training),
     epsilon (the minimum value of the gradient magnitude at which to stop training).
 """
  # add bias to X
  X = add_bias(X)

 # These lines set the size of the GPU thread blocks and the number of blocks needed based on the size of the input data X
  block_size = 1024
  grid_size = (X.size + block_size - 1) // block_size

  # This initializes the weights of the logistic regression model to zeros, with the first weight set to 1 this correspond to bais.
  w = np.zeros(X.shape[1],dtype=float).reshape(1,X.shape[1])
  w[0][0]=1
  # These lines transfer the data y and w to the GPU device.
  y_d = cuda.to_device(y)
  w_d =  cuda.to_device(w)



  grad_loss_d = cuda.to_device(w.copy())

  while True:
    w = w_d.copy_to_host()
    X_d1 = cuda.to_device(X.copy())
    res =np.zeros(X.shape[0],dtype=float).reshape(1,X.shape[0])
    res_d = cuda.to_device(res)
    X2 =  X_d1.T

    vector_matrix_mul[grid_size, block_size](w_d, X2)
    cuda.synchronize()

    matrix_col_sum[grid_size, block_size](X2, res_d)
    cuda.synchronize()

    sigmoid[grid_size, block_size](X_d1,res_d)
    cuda.synchronize()

    substract[grid_size, block_size](y_d,res_d)
    cuda.synchronize()


    X3 = cuda.to_device(X.copy())
    X3 = X3
    vector_matrix_mul[grid_size, block_size](res_d, X3)
    cuda.synchronize()

    matrix_col_sum[grid_size, block_size](X3, grad_loss_d)
    cuda.synchronize()

    # This updates the weights using the gradient and the learning rate
    grad_loss= grad_loss_d.copy_to_host()* (learning_rate/X.shape[0])
    grad_loss_d =  cuda.to_device(grad_loss)

    substract[grid_size, block_size](grad_loss_d,w_d)
    cuda.synchronize()



    # This calculates the magnitude of the gradient vector using the L2 norm
    grad_norm  = np.array([[0]],dtype=float)
    grad_norm_d = cuda.to_device(grad_norm)
    norm2[grid_size, block_size](grad_loss_d,grad_norm_d)
    cuda.synchronize()

    # This checks whether the gradient magnitude is smaller than the epsilon threshold. If so, the function returns the final weight vector. If not, the training loop continues.
    grad_norm = grad_norm_d.copy_to_host()

  
    if math.sqrt(grad_norm[0][0])<= epsilon or np.linalg.norm(w-w_d.copy_to_host())<=epsilon :
      return  w_d.copy_to_host()

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

    # These lines set the size of the GPU thread blocks and the number of blocks needed based on the size of the input data X
    block_size = 200
    grid_size = (X.size + block_size - 1) // block_size

    X_d = cuda.to_device(X)
    w_d = cuda.to_device(w)
    res =np.zeros(X.shape[0],dtype=float).reshape(1,X.shape[0])
    res_d = cuda.to_device(res)
    X2 =  X_d.T

    vector_matrix_mul[grid_size, block_size](w_d, X2)
    cuda.synchronize()

    matrix_col_sum[grid_size, block_size](X2, res_d)
    cuda.synchronize()

    sigmoid[grid_size, block_size](X_d,res_d)
    cuda.synchronize()

    # Round the predictions to the nearest integer (0 or 1)
    predictions = np.round(res_d.copy_to_host())

    return predictions
