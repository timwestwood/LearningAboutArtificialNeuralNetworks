from neuron import neuron
import numpy as np
import matplotlib.pyplot as plt

class network:

    # A simple network with a single hidden layer which consists of 3 neurons.

    def __init__(self, num_inputs):

        self.num_inputs = num_inputs

        self.h1 = neuron(np.random.randn(num_inputs), np.random.randn())
        self.h2 = neuron(np.random.randn(num_inputs), np.random.randn())
        self.h3 = neuron(np.random.randn(num_inputs), np.random.randn())

        self.out = neuron(np.random.randn(3), np.random.randn())

    def output(self, inputs):

        o1 = self.h1.output(inputs)
        o2 = self.h2.output(inputs)
        o3 = self.h3.output(inputs)

        return self.out.output(np.array([o1, o2, o3]))

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

        fig = plt.figure()

        for epoch in range(epochs):

            # Stochastic gradient descent
            gen = np.random.default_rng()
            for n in gen.permutation(true_vals.size):

                x = data[n,:]
                tv = true_vals[n]

                ## Feed forward:
                hidden_layer_output = np.array([self.h1.output(x), self.h2.output(x), self.h3.output(x)])
                predicted_vals[n] = self.out.output(hidden_layer_output) # = self.output(x), but without redoing the hidden layer calculations.

                ## Back propogate:
                fac = learning_rate * self.loss_deriv(predicted_vals[n], tv) # = learning_rate * d_L_d_pred

                # Output layer variables:

                self.out.bias -= fac * self.out.output_deriv(hidden_layer_output, 0)

                for m in range(self.out.weights.size):
                    self.out.weights[m] -= fac * self.out.output_deriv(hidden_layer_output, m+1)

                # Hidden layer variables:

                h1_fac = fac * self.out.output_deriv(hidden_layer_output, self.out.weights.size + 1) # = learning_rate * d_L_d_pred * d_pred_d_h1
                self.h1.bias -= h1_fac * self.h1.output_deriv(x, 0)
                for m in range(self.h1.weights.size):
                    self.h1.weights[m] -= h1_fac * self.h1.output_deriv(x, m+1)

                h2_fac = fac * self.out.output_deriv(hidden_layer_output, self.out.weights.size + 2)
                self.h2.bias -= h2_fac * self.h2.output_deriv(x, 0)
                for m in range(self.h2.weights.size):
                    self.h2.weights[m] -= h2_fac * self.h2.output_deriv(x, m+1)

                h3_fac = fac * self.out.output_deriv(hidden_layer_output, self.out.weights.size + 3)
                self.h3.bias -= h3_fac * self.h3.output_deriv(x, 0)
                for m in range(self.h3.weights.size):
                    self.h3.weights[m] -= h3_fac * self.h3.output_deriv(x, m+1)

            if epoch%10 == 0:

                curr_loss = self.loss(predicted_vals, true_vals)
                print("epoch:", epoch,", loss:", curr_loss)
                plt.plot(epoch, curr_loss, 'ko')

        plt.show()



