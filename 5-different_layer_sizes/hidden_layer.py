from neuron import neuron
import numpy as np

class hidden_layer:

    def __init__(self, num_neurons, inputs_per_neuron):

        self.num_neurons = num_neurons

        self.neurons = []

        for n in range(num_neurons):
            self.neurons.append(neuron(inputs_per_neuron))

    def output(self, inputs):

        out = np.zeros(self.num_neurons)

        for n in range(self.num_neurons):
            out[n] = self.neurons[n].output(inputs)

        return out