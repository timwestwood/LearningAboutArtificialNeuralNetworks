import numpy as np

class neuron:

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def activation_fun(self, x):
        # The logistic function
        return 1.0/(1.0 + np.exp(-x))
    
    def weighted_combination(self, inputs):
        return self.bias + np.dot(inputs, self.weights)
    
    def output(self, inputs):
        return self.activation_fun(self.weighted_combination(inputs))
    
    def activation_fun_deriv(self, inputs):
        # Returns the derivative of the activation function with respect to its scalar input, f',
        # evaluated at the weighted combination of the inputs. This is required for back propogation.
        f = self.output(inputs)
        return f*(1.0 - f) # The logistic function satisfies f'(x) = f(x)*(1 - f(x))

    def weighted_combination_deriv(self, inputs, id):
        # The derivative of the neuron's weighted combination of the inputs with respect to the id-th variable;
        # 0 <= id <= self.weights.size corresponds to the weight derivatives and 
        # self.weights.size <= id < 2*self.weights.size corresponds to the input derivatives.
        # The bias is ignored because the output will always be 1.

        # N.B. The total derivative of the neuron's output with respect to a given variable is given by
        # the product of self.activation_fun_deriv(inputs) and the output of this function. These two parts
        # are calculated separately because the former doesn't depend on id and hence can be re-used.

        if id < self.weights.size:

            return inputs[id]

        else:

            return self.weights[id - self.weights.size]