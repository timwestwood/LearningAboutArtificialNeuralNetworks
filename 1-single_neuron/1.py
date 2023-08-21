import numpy as np

def activation_fun(x):
    # The logistic function
    return 1.0/(1.0 + np.exp(-x))

class neuron:

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def output(self, inputs):
        return activation_fun(self.bias + np.dot(inputs, self.weights))

num_inputs = 5

weights = np.random.randn(num_inputs)
bias = np.random.randn(1)
n = neuron(weights, bias)

x = np.random.randn(num_inputs)
print(n.output(x))