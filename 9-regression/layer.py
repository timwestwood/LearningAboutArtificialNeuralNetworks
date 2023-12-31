from neuron import neuron
import numpy as np
from abc import ABC, abstractmethod

class dense_layer(ABC):

    def __init__(self, num_neurons, inputs_per_neuron):
        self.num_neurons = num_neurons
        self.neurons = []
        for n in range(num_neurons):
            self.neurons.append(neuron(inputs_per_neuron))

    def weighted_combination(self, inputs):
        out = np.zeros(self.num_neurons)
        for n in range(self.num_neurons):
            out[n] = self.neurons[n].output(inputs)
        return out


    def output(self, inputs):
        return self.activation(self.weighted_combination(inputs))
    
    @abstractmethod
    def activation(self, x):
        pass

    @abstractmethod
    def activation_deriv(self, x):
        pass

    def backprop(self, input, d_L_d_output):
        
        a_prime = self.activation_deriv(self.weighted_combination(input)) * d_L_d_output

        # Produce derivatives to propogate backwards
        # N.B. This should be done BEFORE the weights are updated -- it should involve the same weight values as the forward pass.
        d_L_d_input = np.zeros(input.shape)

        for n in range(input.size):
            for k in range(self.num_neurons):
                d_L_d_input[n] += a_prime[k] * self.neurons[k].output_deriv(input, n + input.size)

        # Update weights
        for n in range(self.num_neurons):
            self.neurons[n].bias -= a_prime[n]
            for k in range(self.neurons[n].weights.size):
                self.neurons[n].weights[k] -= a_prime[n] * self.neurons[n].output_deriv(input, k)

        return d_L_d_input
    


class relu_layer(dense_layer):

    def activation(self, x):
        return x * (x>0) # Element-wise multiplication.
    
    def activation_deriv(self, x):
        return 1 * (x>0)


class basic_layer(dense_layer):

    def activation(self, x):
        return x
    
    def activation_deriv(self, x):
        return np.ones(x.shape)




