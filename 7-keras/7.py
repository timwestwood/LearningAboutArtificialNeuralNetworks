from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical # The labels will come as values in [0,9]. This turns them into 'one-hot' arrays.
import mnist
import numpy as np

# Load the data for training and testing
training_ims = mnist.train_images()
training_labels = to_categorical(mnist.train_labels())
test_ims = mnist.test_images()
test_labels = to_categorical(mnist.test_labels())

# Flatten, centre and scale the images (currently they're black and white with pixel values in [0, 255])
training_ims = training_ims.reshape((-1, 784)) # Passing -1 to reshape puts all remaining data into that dimension; i.e. it's equivalent to reshape((training_ims.size/784, 784))
test_ims = test_ims.reshape((-1, 784))
training_ims = (training_ims / 255) - 0.5
test_ims = (test_ims / 255) - 0.5

# Build the network
network = Sequential([
    Dense(89, activation='sigmoid', input_shape=(784,)), # The images are 28x28 flattened to 784x1. The output has 10 classes, so try sqrt(28*28*10) \approx 89 neurons.
    Dense(10, activation='softmax')
])

network.compile(
    optimizer='adam', # A very popular optimizer.
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the network
network.fit(training_ims,
            training_labels,
            epochs=10,
            batch_size=32 # The number of samples to use per update. This is always 1 in the previous hand-written examples.
)

# Test the network
network.evaluate(test_ims, test_labels) # An all-in-one check of the test data.

prediction = network(test_ims[0][None]) # Check a single example to look at the output probabilities. N.B. Because it expects to operate on batches, we have to add this fake 'batch dimension' to make a single sample work. Also, when making batch predictions it scaled better to use network.predict(); e.g. network.predict(test_ims[0:10]).
print("Output probabilities:", prediction.numpy(), "True probabilities:", test_labels[0]) # Cast the tensorflow Tensor to a numpy array for less cluttered printing.

# Save the trained weights for future use
network.save_weights('example7.h5')