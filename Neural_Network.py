from tensorflow import keras
from keras.datasets import mnist
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Load dataset and print shapes of train sets and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Plot a set of data to test
some_digit = x_train[1]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image)
plt.axis("off")

plt.show()

# Flatten and normalize data
x_train = x_train.reshape(x_train.shape[0], 784)
x_train = keras.utils.normalize(x_train, axis=1)
x_train = x_train.astype('float32') / 255

x_test = x_test.reshape(x_test.shape[0], 784)
x_test = keras.utils.normalize(x_test, axis=1)
x_test = x_test.astype('float32') / 255

print(x_train.shape[0], x_train.shape[1])
print(x_test.shape[0], x_test.shape[1])

# Represent output as one-hot vector
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# Training parameters
batch_size = 128
dropout = 0.45


# Building logistic neural network
def logistic_model():
    model = keras.Sequential()  # create a sequential model

    # Define input and hidden layers
    model.add(keras.layers.Dense(256, input_dim=784))    # input layer (784 nodes)
    model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(100, activation='relu'))   # hidden layer 1 (100 nodes)
    model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(100, activation='relu'))   # hidden layer 2 (100 nodes)
    model.add(keras.layers.Dropout(dropout))

    # Define output layer
    model.add(keras.layers.Dense(10, activation='softmax'))   # units (nodes) = 10 for 10 classes

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Training the data by fitting to the NN
def train_model(model):
    history = model.fit(x_train, y_train, epochs=10, batch_size=batch_size, shuffle=True)

    epochs = history.epoch  # track training progress at each epoch
    hist = history.history

    return epochs, hist


if __name__ == '__main__':
    my_model = logistic_model()
    epochs, hist = train_model(my_model)

    loss, acc = my_model.evaluate(x_test, y_test, batch_size=batch_size)
    print("\nTest Accuracy: ", (100*acc))


# Calc classification error for each digit and average classification error for all digits
# ratio of incorrect classification to total number of images for that digit

predictions = my_model.predict(x_test)  # (10000 x 10)

classification_results = np.zeros(10000)
for i in range(x_test.shape[0]):
    classification_results[i] = np.argmax(predictions[i])
    incorrect_predictions = 0

#   print(classification_results[10])  # test prediction - output 0
#   plt.imshow(x_test[10].reshape(28, 28))      # plot 0
#   plt.show()
    for j in range(10):
        if np.logical_and(classification_results[i] == j, y_test[i] == 0):
            incorrect_predictions += 1
        if np.logical_and(classification_results[i] != j, y_test[i] == 1):
            incorrect_predictions += 1

    classification_error = incorrect_predictions/sum()




