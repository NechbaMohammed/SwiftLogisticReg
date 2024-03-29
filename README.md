# SwiftLogisticReg: Accelerated Logistic Regression Package for High-Performance Computing
[![PyPI - Downloads](https://img.shields.io/pypi/dm/SwiftLogisticReg)](https://pypi.org/project/SwiftLogisticReg/)
[![PyPI](https://img.shields.io/pypi/v/SwiftLogisticReg)](https://pypi.org/project/SwiftLogisticReg/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## What is it?

**SwiftLogisticReg** is a package that offers efficient implementations of logistic regression using high-performance computing techniques, with support for both CPU and GPU architectures. The algorithms are implemented in Python 3.8, and the GPU utilization is enhanced through CUDA programming, significantly accelerating the training process.

## Table of Contents
- [Main Features](#main-features)
- [Where to get it](#where-to-get-it)
- [Dependencies](#dependencies)
- [Documentation](#documentation)
- [Authors and Contributors](#authors-and-contributors)

## Main Features

- `logistic_cpu`: Implementation of logistic regression using multi-core parallelism on the CPU. It is designed to efficiently handle large datasets and perform binary classification tasks.

- `logistic_gpu`: Implementation of logistic regression using CUDA programming to harness the power of modern GPUs. The GPU implementation aims to expedite the training process, particularly for big data scenarios.

## Where to get it
The source code is currently hosted on GitHub at:
https://github.com/NechbaMohammed/SwiftLogisticReg

Binary installers for the latest released version are available at the [Python
Package Index (PyPI)](https://pypi.org/project/SwiftLogisticReg)

```sh
# PyPI
pip install SwiftLogisticReg
```

## Dependencies
- [NumPy - Adds support for large, multi-dimensional arrays, matrices and high-level mathematical functions to operate on these arrays](https://www.numpy.org)
- [CUDA Toolkit - The CUDA Toolkit provides support for parallel computing on compatible GPUs, and it is required for utilizing CUDA acceleration in the package.](https://developer.nvidia.com/cuda-downloads)
- [math - Provides various mathematical operations and functions, including basic arithmetic, trigonometry, logarithms, and more.](https://docs.python.org/3/library/math.html)

## Documentation

This documentation provides information about GPU and CPU usage, data description, and performance comparison between different versions of logistic regression.

### GPU and CPU Information


#### GPU Information

| Index | Name      | Memory.Total [MiB] | Memory.Used [MiB] | Memory.Free [MiB] | Temperature.GPU |
|-------|-----------|--------------------|-------------------|-------------------|----------------|
| 0     | Tesla T4  | 15360 MiB          | 0 MiB             | 15101 MiB         | 58             |

#### CPU Information

- Current CPU frequency: 2199.998 MHz
- Minimum CPU frequency: 0.0 MHz
- Maximum CPU frequency: 0.0 MHz
- Number of CPU cores: 2

#### CPU Usage per Core

- Core 0: 60.8%
- Core 1: 62.7%

### Data Description
The [HIGGS dataset](http://archive.ics.uci.edu/dataset/280/higgs) was originally introduced in a research paper titled "Discovering the Higgs boson in the noise" by Baldi et al. (Nature Communications, 2014). The authors of the paper are Pierre Baldi, Peter Sadowski, and Daniel Whiteson. The dataset is used for searching for exotic particles in high-energy physics with deep learning.
![logo](fig/fig1.png)

#### Reference:
- Paper: [Baldi, Pierre, Peter Sadowski, and Daniel Whiteson. "Searching for exotic particles in high-energy physics with deep learning." Nature communications 5, no. 1 (2014): 4308](https://scholar.google.com/scholar_lookup?arxiv_id=1402.4735)
- Dataset: [HIGGS dataset](http://archive.ics.uci.edu/dataset/280/higgs) 

## Load data:
```python	
import numpy as np
import pandas as pd

df  =  pd.read_csv("./data/HIGGS_2M_Row.csv")

y= df['label']
X = df.drop('label',axis=1)
X = X.to_numpy()
y = y.to_numpy().reshape(1,y.shape[0])
```
### Logistic Regression GPU-version
```python
from SwiftLogisticReg.logistic_gpu import LogisticRegressionGPU
from sklearn.metrics import f1_score
import time

# Measure the execution time of the logistic_regression function
start_time = time.time()
model = LogisticRegressionGPU()
model.fit(X, y)
end_time = time.time()

# Print the execution time
print("Execution time:", end_time - start_time, "seconds")

# Use the trained model to make predictions on the training data
y_pred = model.predict(X)

# Calculate and print F1 score
f1Score = f1_score(y_pred[0], y[0])

print("f1_score is", f1Score)
```
#### Results:
```Bash
Execution time: 9.497852802276611 seconds
f1_score is 0.6741570975215551
```

### Logistic Regression CPU-version

```python
from SwiftLogisticReg.logistic_cpu import LogisticRegression 
from sklearn.metrics import accuracy_score, f1_score
import time

# Measure the execution time of the logistic_regression function
start_time = time.time()

# Create an instance of the LogisticRegression class
log_reg = LogisticRegression()

# Train the model on the training data
log_reg.fit(X, y)

end_time = time.time()

predictions = log_reg.predict(X)
# Print the execution time
print("Execution time:", end_time - start_time, "seconds")


# Reshape predictions to match the expected shape for accuracy_score and f1_score
predictions = predictions.reshape(1, predictions.shape[0])

# Calculate F1 score
f1Score = f1_score(predictions[0], y[0])

# Print the results
print("f1_score is", f1Score)
```
#### Results:
```Bash
Execution time: 19.423811674118042 seconds
f1_score is 0.6689305175558841
```
### Logistic Regression Sklearn-version

```python
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Create a LogisticRegression classifier with specified parameters
clf = LogisticRegression(l1_ratio=0.2, tol=2e-2)

# Start timer for model training
t1 = time.time()

# Fit the classifier to the data
clf.fit(X, y[0])

# End timer for model training
t2 = time.time()
print("The execution time: ", t2 - t1)

# Make predictions on the data
y_pred = clf.predict(X)

# Calculate f1 score
f1Score = f1_score(y_pred, y[0])

# Print the results
print("F1 score: ", f1Score)
```
#### Results:

```Bash
The execution time:  22.600391149520874
F1 score:  0.6869399278895209
```
## Authors and Contributors

👤 **Mohammed Nechba**

* LinkedIn: [@NechbaMohammed](https://www.linkedin.com/in/mohammed-nechba-926214225/)
* GitHub: [@NechbaMohammed](https://www.github.com/NechbaMohammed)

👤 **Mohamed Mouhajir**

* LinkedIn: [@mohamedmouhajir](https://www.linkedin.com/in/mohamed-mouhajir-90450a235/)
* GitHub: [@mohamedmouhajir](https://github.com/mohamedmohamed2021)

👤 **Yassine Sedjari**

* LinkedIn: [@yassinesedjari](https://www.linkedin.com/in/yassine-sedjari-4074aa189/)
* GitHub: [@yassinesedjari](https://github.com/Heyyassinesedjari)

[Go to Top](#table-of-contents)