import numpy as np
from abc import ABC, abstractmethod

class dense_layer(ABC):

    def __init__(self, out_size, in_size, is_recurrent=False):

        self.W = np.random.randn(out_size, in_size) / in_size
        self.b = np.random.randn(out_size, 1)
        self.is_recurrent = is_recurrent

        if is_recurrent:
            self.dW = np.zeros(self.W.shape)

    def weighted_combination(self, x):
        return self.W@x + self.b # @ is the shorthand symbol for numpy's matrix multiplication.


    def output(self, x):
        return self.activation(self.weighted_combination(x))
    
    @abstractmethod
    def activation(self, x):
        pass

    @abstractmethod
    def activation_deriv(self, x):
        pass

    def prepare_for_backprop(self):
        if self.is_recurrent:
            self.dW.fill(0.0)

    def update(self):
        if self.is_recurrent:
            self.W -= self.dW

    def backprop(self, input, d_L_d_output):

        d_L_d_output = d_L_d_output[-self.b.size:] # The back-propagated gradient may contain (useless) information related to the sequence input.
        
        a_prime = self.activation_deriv(self.weighted_combination(input)) * d_L_d_output

        # Produce derivatives to propagate backwards.
        # N.B. This should be done BEFORE the weights are updated -- it should involve the same values as the forward pass.
        # Because a single hidden layer is involved in multiple, sequential stages of back-propagation in an RNN, the updates
        # are cached in self.dW and applied later in self.update() if required. Otherwise, the weights are simply updated
        # after this is calculated.
        d_L_d_input = np.transpose(self.W) @ a_prime

        # Update weights.
        self.b -= a_prime

        if self.is_recurrent:
            self.dW += np.outer(a_prime, input)
        else:
            self.W -= np.outer(a_prime, input)

        return d_L_d_input
    


class logistic_layer(dense_layer):

    def activation(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def activation_deriv(self, x):
        f = self.activation(x)
        return f * (1.0 - f)


class basic_layer(dense_layer):

    def activation(self, x):
        return x
    
    def activation_deriv(self, x):
        return np.ones(x.shape)




