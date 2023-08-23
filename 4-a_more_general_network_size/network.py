from neuron import neuron
import numpy as np
import matplotlib.pyplot as plt

class network:

    # A network with an arbitrary number of hidden layers, all of which consists of another arbitrary number of neurons.
    #
    # N.B. It seems that there are almost no practical situations in which one wants more than a single hidden layer
    # (and no theoretical situation in which you could need more than two) and that a good rule of thumb for the number
    # of neurons in this layer is somewhere between the number of inputs and outputs (maybe their arithmetic or geometric mean?
    # -- see https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw).
    # The intention of introducing this generality is just to make sure I understand backpropogation properly.

    def __init__(self, num_inputs, num_hidden_layers, neurons_per_layer):

        self.num_inputs = num_inputs
        self.num_hidden_layers = num_hidden_layers
        self.neurons_per_layer = neurons_per_layer

        self.hidden = [] # Stores all neurons in all hidden layers in 'layer major' order

        # First hidden layer (i.e. the one that sees the input)
        for n in range(neurons_per_layer):
            self.hidden.append(neuron(np.random.randn(num_inputs), np.random.randn()))

        # The 'interior' hidden layers
        for n in range(1, num_hidden_layers):
            for m in range(neurons_per_layer):
                self.hidden.append(neuron(np.random.randn(neurons_per_layer), np.random.randn()))

        # The output neuron/layer
        self.out = neuron(np.random.randn(neurons_per_layer), np.random.randn())






    def output(self, inputs):

        h1 = np.zeros(self.neurons_per_layer)

        for n in range(self.neurons_per_layer):
            h1[n] = self.hidden[n].output(inputs)

        h2 = np.zeros(self.neurons_per_layer)
        for n in range(1, self.num_hidden_layers):
            for m in range(self.neurons_per_layer):
                h2[m] = self.hidden[m + n*self.neurons_per_layer].output(h1)
            h1 = np.copy(h2)

        return self.out.output(h1)
    





    @staticmethod
    def loss(predicted_vals, true_vals):
        # The mean squared error.
        return ((predicted_vals - true_vals) ** 2).mean()

    @staticmethod
    def loss_deriv(predicted_val, true_val):
        return 2.0*(predicted_val - true_val)
    



    

    def train(self, data, true_vals):

        # Assume true_vals contains num_samples values and data is an array of size num_samples-by-num_inputs.

        epochs = 1000
        learning_rate = 0.1

        predicted_vals = np.zeros(true_vals.size)
        h = np.zeros((self.num_hidden_layers, self.neurons_per_layer))
        delta = np.zeros((self.neurons_per_layer, 1))

        my_colour = np.random.rand(3) # For plotting.

        for epoch in range(epochs):

            # Stochastic gradient descent
            gen = np.random.default_rng()
            for n in gen.permutation(true_vals.size):

                x = data[n,:]
                tv = true_vals[n]

                ## Feed forward:

                # The first hidden layer
                for m in range(self.neurons_per_layer):
                    h[0, m] = self.hidden[m].output(x)

                # The 'interior' hidden layers
                for l in range(1, self.num_hidden_layers):
                    for m in range(self.neurons_per_layer):
                        h[l, m] = self.hidden[m + l*self.neurons_per_layer].output(h[l-1, :])

                # The output layer
                predicted_vals[n] = self.out.output(h[self.num_hidden_layers-1, :]) # = self.output(x), but we've cached the hidden layer calculations.

                ## Back propogate:
                fac = learning_rate * self.loss_deriv(predicted_vals[n], tv) # = learning_rate * d_L_d_pred

                # Output layer variables:
                f_prime = fac * self.out.activation_fun_deriv(h[self.num_hidden_layers-1, :])
                self.out.bias -= f_prime
                for m in range(self.out.weights.size):
                    self.out.weights[m] -= f_prime * self.out.weighted_combination_deriv(h[self.num_hidden_layers-1, :], m)

                # 'Interior' hidden layer variables:
                for m in range(self.neurons_per_layer):
                    delta[m, 0] = f_prime * self.out.weighted_combination_deriv(h[self.num_hidden_layers-1, :], m + self.out.weights.size) # = learning_rate * d_L_d_pred * d_pred_d_h

                for l in range(self.num_hidden_layers-1, 0, -1): # N.B. we do want __stop=0 so that the last layer this accesses is 1; we handle the input-touching layer separately.

                    alpha = np.zeros((self.neurons_per_layer, self.neurons_per_layer))

                    for m in range(self.neurons_per_layer):

                        a_prime = self.hidden[m + l*self.neurons_per_layer].activation_fun_deriv(h[l-1, :])

                        self.hidden[m + l*self.neurons_per_layer].bias -= a_prime * delta[m, 0] # = learning_rate * d_L_d_pred * d_pred_d_h * d_h_d_b

                        for k in range(self.hidden[m + l*self.neurons_per_layer].weights.size):
                            self.hidden[m + l*self.neurons_per_layer].weights[k] -= a_prime * delta[m, 0] * self.hidden[m + l*self.neurons_per_layer].weighted_combination_deriv(h[l-1, :], k) # = learning_rate * d_L_d_pred * d_pred_d_h * d_h_d_w

                        for k in range(self.neurons_per_layer):
                            alpha[k, m] = a_prime * self.hidden[m + l*self.neurons_per_layer].weighted_combination_deriv(h[l-1, :], k + self.hidden[m + l*self.neurons_per_layer].weights.size) # Note the transposed structure.

                    delta = np.matmul(alpha, delta) # = learning_rate * d_L_d_pred * d_pred_d_h, but now for the next (i.e. closer to inputs) layer.

                # The first hidden layer:
                for m in range(self.neurons_per_layer):
                    a_prime = self.hidden[m].activation_fun_deriv(x)
                    self.hidden[m].bias -= a_prime * delta[m, 0]
                    for k in range(self.hidden[m].weights.size):
                        self.hidden[m].weights[k] -= a_prime * delta[m, 0] * self.hidden[m].weighted_combination_deriv(x, k)


            if epoch%10 == 0:

                curr_loss = self.loss(predicted_vals, true_vals)
                print("epoch:", epoch,", loss:", curr_loss)
                plt.plot(epoch, curr_loss, 'o', c=my_colour)
                ax = plt.gca()
                ax.set_yscale('log')
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.show(block=False)
                plt.pause(0.001)

        



