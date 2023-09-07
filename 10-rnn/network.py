from layer import logistic_layer, basic_layer
import numpy as np
import matplotlib.pyplot as plt

class network:

    # A simple example of a recurrent neural network (RNN).
    # It is formed of a single logistic hidden layer and an identity-activation output layer.
    # The network takes a scalar sequence of length N and predicts the (N+1)-th value.

    def __init__(self, N):

        num_neurons = int(np.ceil(np.sqrt(N))) # Should be good enough for testing.
        self.l0 = logistic_layer(num_neurons, num_neurons + N, is_recurrent=True) # Input size depends on output size because of recurrence; part of its input is its output from the previous value in the sequence.
        self.l1 = basic_layer(1, num_neurons)

    def output(self, I, cache=None):

        hidden_layer_output_size = self.l0.W.shape[0]
        prev = np.zeros((hidden_layer_output_size, 1))

        T = I.size # Input sequence length

        for t in range(T):

            x = np.concatenate((np.zeros((T, 1)), prev), axis=0)
            x[t] = I[t]
            prev = self.l0.output(x)

            if cache != None:
                cache.append(prev) # Intermediate outputs need to be cached for back-propagation.

        return self.l1.output(prev)
    




    @staticmethod
    def loss(predicted_vals, true_vals):
        # The mean squared error.
        return np.mean((predicted_vals - true_vals) ** 2)

    @staticmethod
    def loss_deriv(predicted_val, true_val):
        return predicted_val - true_val
    



    

    def print_losses(self, epoch, train_true_vals, train_data, test_true_vals, test_data):

        train_predicted_vals = np.zeros(train_true_vals.shape)
        test_predicted_vals = np.zeros(test_true_vals.shape)

        for n in range(train_true_vals.shape[0]):
            train_predicted_vals[n] = self.output(train_data[n])
    
        train_loss = self.loss(train_predicted_vals, train_true_vals)

        for n in range(test_true_vals.shape[0]):
            test_predicted_vals[n] = self.output(test_data[n])

        test_loss = self.loss(test_predicted_vals, test_true_vals)

        print('epoch: %d | training loss = %.3e | test loss = %.3e' % (epoch, train_loss, test_loss))

        fig = plt.figure('loss_fig')
        plt.plot(epoch, train_loss, 'k.')
        plt.plot(epoch, test_loss, 'b.')
        ax = plt.gca()
        ax.set_yscale('log')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show(block=False)
        plt.pause(0.001)







    def train(self, train_data, train_true_vals, test_data, test_true_vals):

        epochs = 300
        learning_rate = 1e-3
        bptt_trunc = 5

        # Find the initial loss
        self.print_losses(0, train_true_vals, train_data, test_true_vals, test_data)

        for epoch in range(epochs):

            # Stochastic gradient descent
            gen = np.random.default_rng()
            for n in gen.permutation(train_true_vals.shape[0]):

                ## Feed forward:
                I = train_data[n]
                h = []
                predicted_val = self.output(I, h)

                ## Truncated back-propagation through time:
                d_L_d_h1 = learning_rate * self.loss_deriv(predicted_val, train_true_vals[n])
                d_L_d_h0 = self.l1.backprop(h[-1], d_L_d_h1)

                self.l0.prepare_for_backprop()
                T = I.size 
                for t in range(T-1, max(0, T-1-bptt_trunc), -1):
                    x = np.concatenate((np.zeros((T, 1)), h[t]), axis=0)
                    x[t] = I[t]
                    d_L_d_h0 = self.l0.backprop(x, d_L_d_h0)

                self.l0.update()

            # Evaluate the new loss
            if epoch%10 == 9:
                self.print_losses(epoch+1, train_true_vals, train_data, test_true_vals, test_data)




