from layer import conv33, max_pool_layer, softmax_layer
import numpy as np
import matplotlib.pyplot as plt

class network:

    # An example of a convolutional neural network (CNN). It is formed of a single convolution layer
    # followed by a single 'max pooling' layer and then a softmax output layer.

    def __init__(self):

        self.l0 = conv33(5) # Use 5 filters.
        self.l1 = max_pool_layer((2,2)) # Take maximums over a moving 2x2 tile.
        self.l2 = softmax_layer(10, 845) # Needs 10 neurons because there are 10 possible output classes.

        # N.B. This network is designed for use on the hand-written digit images from the MNIST database.
        # These images are 28x28. After the convolution with valid padding and 5 3x3 filters, this will become
        # 26x26x5. The max pooling will reduce this to 13x13x5 and hence each neuron in the softmax layer
        # takes 845 inputs.





    def output(self, I):

        return self.l2.output(self.l1.output(self.l0.output(I)).flatten())
    




    @staticmethod
    def loss(predicted_vals, true_vals):
        # The mean cross-entropy loss.
        predicted_vals = np.clip(predicted_vals, 1e-10, 1.0 - 1e-10) # Avoid taking the logarithm of 0.
        loss = -np.mean(np.sum(true_vals * np.log(predicted_vals), axis=1)) # Element-wise multiplication.

        pv_ids = np.argmax(predicted_vals, axis=1)
        tv_ids = np.argmax(true_vals, axis=1)
        acc = np.sum(pv_ids == tv_ids)/pv_ids.size

        return loss, acc # Element-wise multiplication.

    @staticmethod
    def loss_deriv(predicted_val, true_val):
        # This returns the derivative of the single-sample cross-entropy loss with respect to the PRE-softmax
        # output of the neurons in the output layer. Not only is it trivial to evaluate in its own right,
        # but it avoids having to evaluate the Jacobian of softmax.
        # N.B. This expression depends on the true values having the usual 'one-hot' form (well technically only that they are arrays of values summing to 1 -- the point is that they can't be e.g. integer labels).
        return predicted_val - true_val
    



    

    def print_losses(self, epoch, train_true_vals, train_predicted_vals, test_data, test_true_vals, test_predicted_vals):

        # Assumes the predicted values for the training data have already been calculated and stored in train_predicted_vals.
    
        train_loss, train_acc = self.loss(train_predicted_vals, train_true_vals)

        for n in range(test_true_vals.shape[0]):
            test_predicted_vals[n, :] = self.output(test_data[n])

        test_loss, test_acc = self.loss(test_predicted_vals, test_true_vals)

        print('epoch: %d | training loss = %.3f, training accuracy = %.1f%% | test loss = %.3f, test accuracy = %.1f%%' % (epoch, train_loss, train_acc*100, test_loss, test_acc*100))
        plt.plot(epoch, train_loss, 'ko')
        plt.plot(epoch, test_loss, 'bo')
        ax = plt.gca()
        ax.set_yscale('log')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show(block=False)
        plt.pause(0.001)







    def train(self, train_data, train_true_vals, test_data, test_true_vals):

        epochs = 10
        learning_rate = 0.0005

        train_predicted_vals = np.zeros(train_true_vals.shape)
        test_predicted_vals = np.zeros(test_true_vals.shape)

        # Find the initial losses
        for n in range(train_true_vals.shape[0]):
            train_predicted_vals[n, :] = self.output(train_data[n])

        self.print_losses(0, train_true_vals, train_predicted_vals, test_data, test_true_vals, test_predicted_vals)

        for epoch in range(epochs):

            # Stochastic gradient descent
            gen = np.random.default_rng()
            for n in gen.permutation(train_true_vals.shape[0]):

                I = train_data[n]

                ## Feed forward:
                h0 = self.l0.output(I)
                h1 = self.l1.output(h0)
                h1f = h1.flatten()
                train_predicted_vals[n, :] = self.l2.output(h1f) # = self.output(I), but we've cached the layer calculations.

                ## Back propogate:
                f_prime = learning_rate * self.loss_deriv(train_predicted_vals[n, :], train_true_vals[n, :]) # = learning_rate * d_L_d_pre-softmax-prediction

                # Output layer variables:
                for m in range(self.l2.num_nodes):
                    self.l2.nodes[m].bias -= f_prime[m] # = learning_rate * d_L_d_pre-softmax-prediction * d_pre-softmax-prediction_d_b
                    for k in range(self.l2.nodes[m].weights.size):
                        self.l2.nodes[m].weights[k] -= f_prime[m] * self.l2.nodes[m].output_deriv(h1f, k) # = learning_rate * d_L_d_pre-softmax-prediction * d_pre-softmax-prediction_d_wk
                
                # The 'max pool' layer. This doesn't have any weights to update, but we need the derivatives at this layer to propogate backwards.
                d_L_d_h1 = np.zeros(h1.shape)

                for m in range(self.l2.num_nodes):

                    d_z_d_M = np.zeros(self.l2.nodes[m].weights.size)

                    for k in range(self.l2.nodes[m].weights.size):
                        d_z_d_M[k] = self.l2.nodes[m].output_deriv(h1f, k + self.l2.nodes[m].weights.size)

                    d_L_d_h1 += f_prime[m]*d_z_d_M.reshape(h1.shape)

                d_L_d_h0 = self.l1.backprop(h0, d_L_d_h1)
                
                # Finally, the weights in the convolution layer:
                self.l0.backprop(I, d_L_d_h0)

            # Evaluate the new loss
            self.print_losses(epoch+1, train_true_vals, train_predicted_vals, test_data, test_true_vals, test_predicted_vals)




