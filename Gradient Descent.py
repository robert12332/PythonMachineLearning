import numpy as np
import matplotlib.pyplot as plt

# Load data (convert to float)
x_train = np.loadtxt('data.txt', usecols=range(10))  # first 10 columns
y_train = np.loadtxt('data.txt', usecols=10, unpack=True)  # (10K x 1) matrix

# Add x0 = 1 to each row
x_train = np.insert(x_train, 0, 1, axis=1)  # (10K x 11) matrix

# Analytical method
w_analytic = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x_train), x_train)), np.transpose(x_train)), y_train)
print('w_analytic =', w_analytic)

#########################################################################################################
# Gradient Descent

n_sample = x_train.shape[0]  # number of samples = 10K (rows)
n_coeff = x_train.shape[1]  # number of x variables (# of columns)

# Alpha levels
alpha = [0.001, 0.025, 0.005]

# Batch sizes
batch = [n_sample, 10, 15, 20]

loss = dw = dw0 = 0.0

# Initial theta (weights)
w = np.full([11, 1], 0.01)  # column matrix of w (11 x1) (w0 to w10)
w0 = 0.01

def predict_model():    # function to compute predictions: y hat
    for i in range(n_coeff):
        y_hat = np.sum(np.dot(w[i], x_train[i]))


def loss_function(batch_sz):    # function to compute SSE: loss function
    y_hat = predict_model()
    loss = (1 / (2 * batch_sz)) * np.sum(y_hat - y_train) ** 2  # normalized loss function
    return loss


def partial_derivative(batch_sz):   # function to compute error of current w
    y_hat = predict_model()
    for i in range(n_coeff):
        dw = (1/batch_sz) * np.sum(y_hat - y_train) * (x_train[i])
        dw0 = (1/batch_sz) * np.sum(y_hat - y_train)
    return dw, dw0


def create_batches(batch_sz):  # function to create a list containing mini-batches
    mini_batches = []   # list of samples in a mini-batch
    n_minibatches = int(x_train.shape[0] / batch_sz)   # calc number of batches based on batch size

    for i in range(0, n_sample, batch_sz):
        x_minibatch = x_train[i:(i + batch_sz)]      # mini-batch go from sample 0-10K
        y_minibatch = y_train[i:(i + batch_sz)]      # in batch size interval
        mini_batches.append((x_minibatch, y_minibatch))     # mini-batch now contains the samples called above
    return mini_batches


def gradient_descent(x_train, y_train, n_epoch=100):  # function to perform gradient descent
    error_list = []

    for epoch in range(n_epoch):    # loop for each epoch
        if epoch == n_epoch:
            break
        for batch_sz in batch:      # loop for each batch size
            mini_batches = create_batches(batch_sz)
            for samples in mini_batches:     # loop for each mini-batch
                x_minibatch, y_minibatch = samples
                for i in range(n_coeff):
                    y_hat = predict_model()
                    loss = loss_function(batch_sz)
                    dw = partial_derivative(batch_sz)
                    dw0 = partial_derivative(batch_sz)

                    for learning_rate in alpha:     # loop for each alpha
                        w = w[i] - learning_rate * dw * x_train
                        w0 = w0 - learning_rate * dw0
                return w, w0

                error_list.append(loss)


if __name__ == '__main__':
    tol = 0.01
    w = gradient_descent(x_train, y_train, n_epoch=100)
    while np.linalg.norm(0.01-w) > tol:
        w_new, w0_new = gradient_descent(x_train, y_train, n_epoch=100)







