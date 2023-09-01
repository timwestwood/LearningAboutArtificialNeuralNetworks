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
        
        # Update weights
        a_prime = self.activation_deriv(self.weighted_combination(input))

        for n in range(self.num_neurons):
            self.neurons[n].bias -= d_L_d_output[n]
            for k in range(self.neurons[n].weights.size):
                self.neurons[n].weights[k] -= d_L_d_output[n] * a_prime[n] * self.neurons[n].output_deriv(input, k)

        # Produce derivatives to propogate backwards
        d_L_d_input = np.zeros(input.shape)

        for n in range(input.size):
            for k in range(self.num_neurons):
                d_L_d_input[n] = d_L_d_output[k] * a_prime[k] * self.neurons[k].output_deriv(input, n + input.size)

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




