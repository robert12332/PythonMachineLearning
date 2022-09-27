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
n_epoch = 10000
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


# Gradient descent training
for cls in range(k):   # loop for each class
    theta = theta_init
#    loss = np.zeros((n_sample, k))    # initial loss (60000 x 10)
    gradient = np.zeros((n_sample, n_features))  # initial gradient (60000 x 10)
    print("Training for class ", str(cls))
    for epoch in range(n_epoch):        # loop for each epoch
        for i in range(n_sample):       # loop for training data
            if y_train[i] == cls:       # represent labels y in one-hot vector
                y_label = 1
            else:
                y_label = 0

            for j in range(n_features):   # loop for each row
                # loss[i, j] = 1/n_sample * (-(np.sum(y_label * np.log(y_hat) + (1-y_label) * np.log(1 - y_hat))))

                # Gradient
                y_hat = sigmoid_function(np.dot(x_train[i, :], theta[:, cls]))
                gradient[i, j] = 1/n_sample * np.dot((y_hat - y_label), x_train[i, j])

                # Update theta
                theta[:, cls] = theta[:, cls] - alpha * gradient[i, j]

# Testing







