from layer import relu_layer, basic_layer
import numpy as np
import matplotlib.pyplot as plt

class network:

    # An example of a neural network for regression as opposed to classification.
    # It is formed of a single ReLU hidden layer and an identity-activation output layer.

    def __init__(self, num_inputs):

        num_neurons = int(np.ceil(np.sqrt(num_inputs))) # Should be good enough for testing.
        self.l0 = relu_layer(num_neurons, num_inputs)
        self.l1 = basic_layer(1, num_neurons)

    def output(self, I):

        return self.l1.output(self.l0.output(I))
    




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

        print('epoch: %d | training loss = %.3f | test loss = %.3f' % (epoch, train_loss, test_loss))
        plt.plot(epoch, train_loss, 'ko')
        plt.plot(epoch, test_loss, 'bo')
        ax = plt.gca()
        ax.set_yscale('log')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show(block=False)
        plt.pause(0.001)







    def train(self, train_data, train_true_vals, test_data, test_true_vals):

        epochs = 1000
        learning_rate = 1e-3

        # Find the initial loss
        self.print_losses(0, train_true_vals, train_data, test_true_vals, test_data)

        for epoch in range(epochs):

            # Stochastic gradient descent
            gen = np.random.default_rng()
            for n in gen.permutation(train_true_vals.shape[0]):

                I = train_data[n]

                ## Feed forward:
                h0 = self.l0.output(I)
                predicted_val = self.l1.output(h0)

                ## Back propogate:
                d_L_d_h1 = np.array([learning_rate * self.loss_deriv(predicted_val, train_true_vals[n])])
                d_L_d_h0 = self.l1.backprop(h0, d_L_d_h1)
                _ = self.l0.backprop(I, d_L_d_h0)

            # Evaluate the new loss
            if epoch%10 == 9:
                self.print_losses(epoch+1, train_true_vals, train_data, test_true_vals, test_data)




