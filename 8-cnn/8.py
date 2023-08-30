from network import network
import mnist
from keras.utils import to_categorical # The labels will come as values in [0,9]. This turns them into 'one-hot' arrays.
import matplotlib.pyplot as plt

# Load the data for training and testing
training_ims = mnist.train_images()
training_labels = to_categorical(mnist.train_labels())
test_ims = mnist.test_images()
test_labels = to_categorical(mnist.test_labels())

training_ims = (training_ims / 255) - 0.5
test_ims = (test_ims / 255) - 0.5

# Train the network
cnn = network()
cnn.train(training_ims[0:1000], training_labels[0:1000]) # Use a subset of the available training data because this is going to be slow...

# Test the trained network
print("Test output: ", cnn.output(test_ims[0]), "(should be close to", test_labels[0], ")")

plt.show()