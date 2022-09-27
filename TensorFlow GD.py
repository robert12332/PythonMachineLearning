from tensorflow import keras
from keras.datasets import mnist
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Load dataset and print shapes
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print('x_train: ' + str(x_train.shape))     # x_train is (60000, 28x28)
print('y_train: ' + str(y_train.shape))     # y_train holds 60K labels from 0 to 9
print('x_test: ' + str(x_test.shape))       # x_test is (10000, 28x28)
print('y_test: ' + str(y_test.shape))       # y_test is (10000)

# Reshape 28x28 pixels into input vectors
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])       # x_train is (60000 x 784)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])         # x_test is (10000 x 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizing data
x_train /= 255
x_test /= 255

# Insert x0 = 1 to each row
x_train = np.insert(x_train, 0, 1, axis=1)      # x_train is (60000 x 785)

# Parameters for gradient descent
alpha = 0.01    # learning rate
tol = 0.001     # tolerance value
n_epoch = 1000
k = 10          # number of classes
n_sample = x_train.shape[0]     # number of training sets (60000)
n_features = x_train.shape[1]   # number of features (785)

# Initialize theta as 0.01
theta_init = np.full([x_train.shape[1], k], 0.01)  # theta is (785 x 10)


def sigmoid_function(z):     # function to calc sigmoid
    sigmoid = 1/(1 + np.exp(-z))
    return sigmoid


def softmax_function(z):     # function to calc softmax
    softmax = np.zeros((z.shape[0], z.shape[1]))
    for i in range(k):
        softmax = np.exp(z)/np.sum(np.exp(z))
    return softmax


# main section for gradient descent training
for cls in range(k):   # loop for each class
    theta = theta_init
    loss_values = []
    gradient_values = []
    print("Training for class ", str(cls))
    for epoch in range(n_epoch):
        for i in range(n_sample):       # loop for training data
            if y_train[i] == cls:       # represent labels y in one-hot vector
                y_label = 1
            else:
                y_label = 0

            for j in range(n_features):   # loop for each row
                # Loss function
                y_hat = sigmoid_function(np.dot(x_train[i, :], theta[:, cls]))
                cost = 1/n_sample * (-(np.sum(y_label * np.log(y_hat) + (1-y_label) * np.log(1 - y_hat))))
                loss_values.append(cost)

                # Gradient
                gradient = 1/n_sample * np.dot((y_hat - y_label), x_train[i, :])
                # gradient_values.append(gradient)

                # Update theta
                theta[:, cls] = theta[:, cls] - np.dot(alpha, gradient)

        # set stop point for training
        if abs(cost - loss_values[-2]) < tol:
            break

# Plot theta
for cls in range(k):
  plt.imshow(theta[0:784, cls].reshape(28, 28))
  plt.colorbar()
  plt.show()

  print(theta[:, cls])

# Testing section using trained model

for cls in range(k):
    x_test = x_test[:, 0:785]
    correct_predictions = 0
    for i in range(x_test.shape[0]):
        z_test = sigmoid_function(np.dot(x_test[i, :], theta[:, cls]))  # y_hat passed through sigmoid
#       predictions = softmax_function(z_test)
        if y_test == cls:  # represent y_test as one hot vector
            y_test = 1
        else:
            y_test = 0

        # set prediction boundary and calc accurate predictions
        if np.logical_and(z_test >= 0.5, y_test ==1):
            correct_predictions += 1
        if np.logical_and(z_test < 0.5, y_test == 0):
            correct_predictions += 1

    accuracy_percentage = (correct_predictions/x_test.shape[0]) * 100
    print("Accuracy for class ", str(cls), "=", accuracy_percentage)