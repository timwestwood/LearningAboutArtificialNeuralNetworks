import numpy as np

def activation_fun(x):
    # The logistic function
    return 1.0/(1.0 + np.exp(-x))

def activation_fun_deriv(x):
    # The logistic function satisfies f'(x) = f(x)*(1 - f(x))
    f = activation_fun(x)
    return f*(1.0 - f)

class neuron:

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def output(self, inputs):
        return activation_fun(self.bias + np.dot(inputs, self.weights))

    def output_deriv(self, inputs, id):

        # The derivative of the neuron's output with respect to the id-th variable;
        # id = 0 corresponds to the bias, 1 <= id <= self.weights.size corresponds to the weights
        # and self.weights.size + 1 <= id <= 2*self.weights.size corresponds to the input values.

        d = activation_fun_deriv(self.bias + np.dot(inputs, self.weights))

        if id == 0:

            return d

        elif id <= self.weights.size:

            return d*inputs[id-1]

        else:

            return d*self.weights[id - self.weights.size - 1]