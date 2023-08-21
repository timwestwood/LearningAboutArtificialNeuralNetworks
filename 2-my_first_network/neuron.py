import numpy as np

class neuron:

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    @staticmethod
    def activation_fun(x):
        # The logistic function
        return 1.0/(1.0 + np.exp(-x))

    def output(self, inputs):
        return self.activation_fun(self.bias + np.dot(inputs, self.weights))