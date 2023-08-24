from neuron import neuron
import numpy as np
from abc import ABC, abstractmethod
from scipy.special import softmax

class layer(ABC):

    def __init__(self, num_neurons, inputs_per_neuron):

        self.num_neurons = num_neurons

        self.neurons = []

        for n in range(num_neurons):
            self.neurons.append(neuron(inputs_per_neuron))

    @abstractmethod
    def activation(self, x):
        pass

    def output(self, inputs):

        out = np.zeros(self.num_neurons)

        for n in range(self.num_neurons):
            out[n] = self.neurons[n].output(inputs)

        return self.activation(out)
    


class logistic_layer(layer):

    def activation(self, x):
        return np.ones(x.shape) / (1.0 + np.exp(-x)) # Element-wise division.
    
    def activation_deriv(self, inputs):
        f = self.output(inputs)
        return f * (1.0 - f) # Element-wise multiplication.
    


class softmax_layer(layer):

    def activation(self, x):
        return softmax(x)
    
    # The code is simpler if we don't define the activation Jacobian -- see network.loss_deriv