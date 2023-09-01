from network import network
import numpy as np
import matplotlib.pyplot as plt

# Use the data from https://www.kaggle.com/datasets/muhammadtalharasool/simple-gender-classification
# to predict the weight of an individual given their gender, age and height.
training_data = np.array([[0, 32, 175],
                         [0, 25, 182],
                         [1, 41, 160],
                         [0, 38, 178],
                         [1, 29, 165],
                         [0, 45, 190],
                         [1, 27, 163],
                         [0, 52, 179],
                         [1, 31, 168],
                         [0, 36, 177],
                         [1, 24, 162],
                         [0, 44, 183],
                         [1, 28, 166],
                         [0, 29, 181],
                         [1, 33, 170],
                         [0, 37, 176],
                         [1, 26, 169],
                         [0, 28, 182],
                         [0, 33, 178],
                         [1, 44, 160],
                         ], dtype='float64')

training_weights = np.array([70, 85, 62, 79, 58, 92, 55, 83, 61, 76, 53, 87, 60, 84, 65, 78, 59, 75, 82, 58], dtype='float64' )

test_data = np.array([[0, 29, 176],
                      [1, 31, 165],
                      [0, 40, 187],
                      [1, 27, 163],
                      [0, 47, 181],
                      [1, 35, 170],
                      [0, 42, 175],
                      [1, 26, 160],
                      [0, 49, 183],
                      [1, 30, 168],
                    ], dtype='float64')

test_weights = np.array([74, 63, 90, 56, 85, 65, 80, 53, 92, 61], dtype='float64' )

# Normalise the data
data_mean = np.mean(training_data, axis=0)
weights_mean = np.mean(training_weights)

training_data -= data_mean
test_data -= data_mean

training_weights -= weights_mean
test_weights -= weights_mean

data_scale = np.amax(np.abs(training_data), axis=0)
weights_scale = np.amax(np.abs(training_weights))

training_data /= data_scale
test_data /= data_scale

training_weights /= weights_scale
test_weights /= weights_scale

# Train and test the network
n = network(3)
n.train(training_data, training_weights, test_data, test_weights)

plt.show()