

#Question 1

import numpy as np
import matplotlib.pyplot as plt

def logistic_sigmoid(x):
  return 1 / (1 + np.exp(-x))

def logistic_sigmoid_derivative(x):
  return x * (1 - x)

def predict(X, first_weights, second_weights):
  hidden_layer = logistic_sigmoid(np.dot(X, first_weights))
  output_layer = logistic_sigmoid(np.dot(hidden_layer, second_weights))
  return output_layer

def train(X, y, first_weights, second_weights, num_epochs):
  errors = []
  for epoch in range(num_epochs):
    prediction = predict(X, first_weights, second_weights)
    error = prediction - y
    gradient2 = np.dot(prediction.T, error * logistic_sigmoid_derivative(prediction))
    error_hidden_layer = np.dot(error, second_weights.T)
    gradient1 = np.dot(X.T, error_hidden_layer * logistic_sigmoid_derivative(prediction))
    first_weights -= eta * gradient1
    second_weights -= eta * gradient2
    errors.append(np.mean(error**2))
  return first_weights, second_weights, errors

# Generate Parity-3 dataset
X = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
y = np.array([[0], [1], [1], [0], [1], [0], [0], [1]])

eta = 0.05
num_epochs = 2000
num_runs = 100
errors = []

for i in range(num_runs):
  # Initialize weights with i.i.d randomization using standard Gaussian distribution
  first_weights = np.random.normal(0, 1, (3, 3))
  second_weights = np.random.normal(0, 1, (3, 1))
  first_weights_final, second_weights_final, errors_per_run = train(X, y, first_weights, second_weights, num_epochs)
  errors.append(errors_per_run)

# Calculate mean square error for each iteration
mean_errors = np.mean(errors, axis=0)

# Plot graph of mean square error as a function of iteration index
plt.plot(mean_errors)
plt.xlabel('Iteration index')
plt.ylabel('Mean square error')
plt.show()
# -----------------------------------------------

#Question 2

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
  return x * (1 - x)

def predict(X, first_weights, second_weights):
  hidden_layer = sigmoid(np.dot(X, first_weights))
  output_layer = sigmoid(np.dot(hidden_layer, second_weights))
  return output_layer

def train(X, y, weights1, second_weights, num_epochs):
  errors = []
  for epoch in range(num_epochs):
    prediction = predict(X, weights1, second_weights)
    error = prediction - y
    gradient2 = np.dot(prediction.T, error * sigmoid_derivative(prediction))
    error_hidden_layer = np.dot(error, second_weights.T)
    gradient1 = np.dot(X.T, error_hidden_layer * sigmoid_derivative(prediction))
    weights1 -= eta * gradient1
    second_weights -= eta * gradient2
    errors.append(np.mean(error**2))
  return weights1, second_weights, errors

# The parity-3 dataset
X = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
y = np.array([[0], [1], [1], [0], [1], [0], [0], [1]])

eta = 0.05
num_epochs = 2000
num_runs = 100
errors = []

for i in range(num_runs):
  # Initialize weights with i.i.d randomization using standard Gaussian distribution
  first_weights = np.random.normal(0, 1, (3, 6))
  second_weights = np.random.normal(0, 1, (6, 1))
  first_weights_final, second_weights_final, errors_per_run = train(X, y, first_weights, second_weights, num_epochs)
  errors.append(errors_per_run)

# Mean square error for each iteration
mean_errors = np.mean(errors, axis=0)

# Plot graph of mean square error as a function of iteration index
plt.plot(mean_errors)
plt.xlabel('Iteration index')
plt.ylabel('Mean square error')
plt.show()