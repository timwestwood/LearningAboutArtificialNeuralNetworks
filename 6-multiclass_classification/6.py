import numpy as np
from network import network
import matplotlib.pyplot as plt

# This example will try to predict the gender (male = 0, female = 1) given the height (cm)
# and weight (kg) of the individual; i.e. the size of the input is 2. The data is taken from
# the kaggle data set at https://www.kaggle.com/datasets/muhammadtalharasool/simple-gender-classification

data = np.array([[175, 70], [182, 85], [160, 62], [178, 79], [165, 58], [190, 92], [163, 55], [179, 83], [168, 61]])
genders = np.array([[0, 1], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0]])

# Centre the data (to keep data in the region where the sigmoid activation function isn't 'flat'):
# N.B. Most of the time scaling would be useful too, but it doesn't matter much for this example.
training_data_mean = data.mean(0)

# Recreate the network from example 3.
n1 = network(2, 1, [3], 2)
n1.train(data-training_data_mean, genders)

# See how it performs on data not in the training set:
print("Output for [177, 76] is", n1.output(np.array([177, 76])-training_data_mean), "(it should be close to [0, 1]).")
print("Output for [162, 53] is", n1.output(np.array([162, 53])-training_data_mean), "(it should be close to [1, 0]).")

# Try a multi-layer network with different layer sizes.
n2 = network(2, 2, [3, 2], 2)
n2.train(data-training_data_mean, genders)

# See how it performs on data not in the training set:
print("Output for [177, 76] is", n2.output(np.array([177, 76])-training_data_mean), "(it should be close to [0, 1]).")
print("Output for [162, 53] is", n2.output(np.array([162, 53])-training_data_mean), "(it should be close to [1, 0]).")

plt.show()