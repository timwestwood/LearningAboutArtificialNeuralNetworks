from layer import logistic_layer, softmax_layer
import numpy as np
import matplotlib.pyplot as plt

class network:

    # A network with an arbitrary number of hidden layers, each of which comprises an arbitrary number of neurons.
    #
    # N.B. It seems that there are almost no practical situations in which one wants more than a single hidden layer
    # (and no theoretical situation in which you could need more than two) and that a good rule of thumb for the number
    # of neurons in this layer is somewhere between the number of inputs and outputs (maybe their arithmetic or geometric mean?
    # -- see https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw).

    def __init__(self, num_inputs, num_hidden_layers, neurons_per_layer, num_classes):

        self.num_hidden_layers = num_hidden_layers

        self.layers = [] # Stores all hidden layers

        # First hidden layer (i.e. the one that sees the input)
        self.layers.append(logistic_layer(neurons_per_layer[0], num_inputs))

        # The 'interior' hidden layers
        for n in range(1, num_hidden_layers):
            self.layers.append(logistic_layer(neurons_per_layer[n], neurons_per_layer[n-1]))

        # The output layer
        self.out = softmax_layer(num_classes, self.layers[-1].num_neurons)






    def output(self, inputs):

        h1 = self.layers[0].output(inputs)

        for n in range(1, self.num_hidden_layers):
            h2 = self.layers[n].output(h1)
            h1 = np.copy(h2)

        return self.out.output(h1)
    




    @staticmethod
    def loss(predicted_vals, true_vals):
        # The total cross-entropy loss.
        predicted_vals = np.clip(predicted_vals, 1e-10, 1.0 - 1e-10) # Avoid taking the logarithm of 0.
        return -np.sum(true_vals * np.log(predicted_vals)) # Element-wise multiplication.

    @staticmethod
    def loss_deriv(predicted_val, true_val):
        # This returns the derivative of the single-sample cross-entropy loss with respect to the PRE-softmax
        # output of the neurons in the output layer. Not only is it trivial to evaluate in its own right,
        # but it avoids having to evaluate the Jacobian of softmax.
        # N.B. This expression depends on the true values having the usual 'one-hot' form (well technically only that they are arrays of values summing to 1 -- the point is that they can't be e.g. integer labels).
        return predicted_val - true_val
    



    

    def train(self, data, true_vals):

        # Assume true_vals has size num_samples-by-num_classes, and data has size num_samples-by-num_inputs.

        epochs = 1000
        learning_rate = 0.1

        predicted_vals = np.zeros((true_vals.shape[0], self.out.num_neurons))

        my_colour = np.random.rand(3) # For plotting.

        for epoch in range(epochs):

            # Stochastic gradient descent
            gen = np.random.default_rng()
            for n in gen.permutation(true_vals.shape[0]):

                x = data[n,:]

                ## Feed forward:

                h = [] # To cache the output of the hidden layers

                # The first hidden layer
                h.append(self.layers[0].output(x))

                # The 'interior' hidden layers
                for l in range(1, self.num_hidden_layers):
                    h.append(self.layers[l].output(h[l-1]))

                # The output layer
                predicted_vals[n, :] = self.out.output(h[-1]) # = self.output(x), but we've cached the hidden layer calculations.

                ## Back propogate:
                f_prime = learning_rate * self.loss_deriv(predicted_vals[n, :], true_vals[n, :]) # = learning_rate * d_L_d_pre-softmax-prediction

                # Output layer variables:
                for m in range(self.out.num_neurons):
                    self.out.neurons[m].bias -= f_prime[m] # = learning_rate * d_L_d_pre-softmax-prediction * d_pre-softmax-prediction_d_b
                    for k in range(self.out.neurons[m].weights.size):
                        self.out.neurons[m].weights[k] -= f_prime[m] * self.out.neurons[m].output_deriv(h[-1], k) # = learning_rate * d_L_d_pre-softmax-prediction * d_pre-softmax-prediction_d_wk

                # 'Interior' hidden layer variables:
                delta = np.zeros((self.layers[-1].num_neurons, 1))

                for m in range(self.out.num_neurons):
                    for k in range(self.layers[-1].num_neurons):
                        delta[k, 0] = f_prime[m] * self.out.neurons[m].output_deriv(h[-1], k + self.out.neurons[m].weights.size) # = learning_rate * d_L_d_pre-softmax-prediction * d_pre-softmax-prediction_d_h

                for l in range(self.num_hidden_layers-1, 0, -1): # N.B. we do want __stop=0 so that the last layer this accesses is 1; we handle the input-touching layer separately.

                    alpha = np.zeros((self.layers[l-1].num_neurons, self.layers[l].num_neurons))

                    a_prime = self.layers[l].activation_deriv(h[l-1])

                    for m in range(self.layers[l].num_neurons):

                        self.layers[l].neurons[m].bias -= a_prime[m] * delta[m, 0] # = learning_rate * d_L_d_pred * d_pred_d_h * d_h_d_b

                        for k in range(self.layers[l-1].num_neurons):

                            self.layers[l].neurons[m].weights[k] -= a_prime[m] * delta[m, 0] * self.layers[l].neurons[m].output_deriv(h[l-1], k) # = learning_rate * d_L_d_pred * d_pred_d_h * d_h_d_w

                            alpha[k, m] = a_prime[m] * self.layers[l].neurons[m].output_deriv(h[l-1], k + self.layers[l].neurons[m].weights.size) # Note the transposed structure.    

                    delta = np.matmul(alpha, delta) # = learning_rate * d_L_d_pred * d_pred_d_h, but now for the next (i.e. closer to inputs) layer.

                # The first hidden layer:
                a_prime = self.layers[0].activation_deriv(x)
                for m in range(self.layers[0].num_neurons):
                    self.layers[0].neurons[m].bias -= a_prime[m] * delta[m, 0]
                    for k in range(self.layers[0].neurons[m].weights.size):
                        self.layers[0].neurons[m].weights[k] -= a_prime[m] * delta[m, 0] * self.layers[0].neurons[m].output_deriv(x, k)


            if epoch%10 == 0:

                curr_loss = self.loss(predicted_vals, true_vals)
                print("epoch:", epoch," loss =", curr_loss)
                plt.plot(epoch, curr_loss, 'o', c=my_colour)
                ax = plt.gca()
                ax.set_yscale('log')
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.show(block=False)
                plt.pause(0.001)

        



