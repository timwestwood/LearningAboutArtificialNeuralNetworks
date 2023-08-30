import numpy as np

class neuron:

    def __init__(self, num_inputs, weights=None, bias=None):

        # Default arguments are evaluated once at function definition, not at each call, and moreover
        # can't depend on another argument, so we have to do it this way.
        if weights is None:
            weights = np.random.randn(num_inputs) / num_inputs

        if bias is None:
            bias = np.random.randn()

        self.weights = weights
        self.bias = bias
    
    def output(self, inputs):
        return self.bias + np.dot(inputs, self.weights)

    def output_deriv(self, inputs, id):
        # The derivative of the neuron's output with respect to the id-th variable;
        # 0 <= id < self.weights.size corresponds to the weight derivatives and 
        # self.weights.size <= id < 2*self.weights.size corresponds to the input derivatives.
        # The bias is ignored because the output will always be 1.

        # N.B. The total derivative of the neuron's output with respect to a given variable is given by
        # the product of the derivative of the containing layer's activation function with the output of this function.
        # These two parts are calculated separately because the former doesn't depend on id and hence can be re-used.

        if id < self.weights.size:

            return inputs[id]

        else:

            return self.weights[id - self.weights.size]
        