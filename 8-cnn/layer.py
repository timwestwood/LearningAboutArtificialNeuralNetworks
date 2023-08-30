from neuron import neuron
from filter import filter
import numpy as np
from abc import ABC, abstractmethod
from scipy.special import softmax

class layer(ABC):

    def __init__(self, num_nodes, node_dims, node_type):

        self.num_nodes = num_nodes

        self.nodes = []

        for n in range(num_nodes):
            self.nodes.append(node_type(node_dims))

    @abstractmethod
    def activation(self, x):
        pass

    @abstractmethod
    def output(self, inputs):
        pass








    
class dense_layer(layer):

    def __init__(self, num_neurons, inputs_per_neuron):
        super(dense_layer, self).__init__(num_neurons, inputs_per_neuron, neuron)

    def output(self, inputs):

        out = np.zeros(self.num_nodes)

        for n in range(self.num_nodes):
            out[n] = self.nodes[n].output(inputs)

        return self.activation(out)
    


class logistic_layer(dense_layer):

    def activation(self, x):
        return np.ones(x.shape) / (1.0 + np.exp(-x)) # Element-wise division.
    
    def activation_deriv(self, inputs):
        f = self.output(inputs)
        return f * (1.0 - f) # Element-wise multiplication.
    


class softmax_layer(dense_layer):

    def activation(self, x):
        return softmax(x)
    
    # The code is simpler if we don't define the activation Jacobian -- see network.loss_deriv









class conv_layer(layer):

    def __init__(self, num_filters, num_rows, num_cols):
        super(conv_layer, self).__init__(num_filters, (num_rows, num_cols), filter)

    def activation(self, x):
        return x
    
    def output(self, I):

        im = self.nodes[0].apply(I)

        filtered_ims = np.zeros(im.shape + (self.num_nodes,)) # Tuple concatenation.
        filtered_ims[:, :, 0] = im

        for n in range(1, self.num_nodes):
            filtered_ims[:, :, n] = self.nodes[n].apply(I)

        return filtered_ims
    
    def backprop(self, I, D):

        for n in range(self.num_nodes):
            self.nodes[n].backprop(I, D[:, :, n])

    

class conv33(conv_layer):

    def __init__(self, num_filters):
        super(conv33, self).__init__(num_filters, 3, 3)
    







class pool_layer:

    def __init__(self, dims):
        self.pool_height = dims[0]
        self.pool_width = dims[1]

    @abstractmethod
    def pooling_fun(self, I):
        pass

    def output(self, I):

        h_fac = I.shape[0] // self.pool_height
        w_fac = I.shape[1] // self.pool_width

        out = np.zeros((h_fac, w_fac, I.shape[2]))

        for im in range(I.shape[2]):
            for row in range(h_fac):
                for col in range(w_fac):
                    out[row, col, im] = self.pooling_fun(I[row*self.pool_height:(row+1)*self.pool_height, col*self.pool_width:(col+1)*self.pool_width, im])

        return out
    

class max_pool_layer(pool_layer):

    def pooling_fun(self, I):
        return np.amax(I, axis=(0,1))
    
    def backprop(self, I, D):

        h_fac = I.shape[0] // self.pool_height
        w_fac = I.shape[1] // self.pool_width

        out = np.zeros(I.shape)

        for im in range(I.shape[2]):
            for row in range(h_fac): # row of the pooled image, which this doesn't form
                for col in range(w_fac): # column of the pooled image, which this doesn't form
                    Isub = I[row*self.pool_height:(row+1)*self.pool_height, col*self.pool_width:(col+1)*self.pool_width, im]
                    max_val = np.amax(Isub, axis=(0,1))
                    for i in range(self.pool_height):
                        for j in range(self.pool_width):
                            if (Isub[i, j] == max_val):
                                out[row*self.pool_height + i, col*self.pool_width + j, im] = D[row, col, im]

        return out
