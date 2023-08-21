from neuron import neuron
import numpy as np

class network:

    # A simple network with a single hidden layer which consists of 5 neurons.

    def __init__(self, num_inputs):

        self.h1 = neuron(np.random.randn(num_inputs), np.random.randn())
        self.h2 = neuron(np.random.randn(num_inputs), np.random.randn())
        self.h3 = neuron(np.random.randn(num_inputs), np.random.randn())
        self.h4 = neuron(np.random.randn(num_inputs), np.random.randn())
        self.h5 = neuron(np.random.randn(num_inputs), np.random.randn())

        self.out = neuron(np.random.randn(5), np.random.randn())

    def output(self, inputs):

        o1 = self.h1.output(inputs)
        o2 = self.h2.output(inputs)
        o3 = self.h3.output(inputs)
        o4 = self.h4.output(inputs)
        o5 = self.h5.output(inputs)

        return self.out.output(np.array([o1, o2, o3, o4, o5]))



num_inputs = 10

inputs = np.random.randn(num_inputs)

n = network(num_inputs)

print(n.output(inputs))