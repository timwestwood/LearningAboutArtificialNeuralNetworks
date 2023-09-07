from network import network
import numpy as np
import matplotlib.pyplot as plt

# Define the curve whose values we want to be able to predict.
theta = np.linspace(0.0, 3.0*np.pi, 400)

def fun(theta):
    return (1.0 - theta/(6.0 * np.pi))*np.sin(theta)

curve = fun(theta) # This already oscillates about 0 with magnitude less than 1, so we won't normalise the data.

# Assemble the data
seq_length = 50

num_seqs = theta.size - seq_length
data = np.zeros((num_seqs, seq_length))
true_vals = np.zeros(num_seqs)
for i in range(num_seqs):
    data[i, :] = curve[i:i+seq_length]
    true_vals[i] = curve[i+seq_length]

# Shuffle the data and divide into training and testing/validation sets
gen = np.random.default_rng()
I = gen.permutation(num_seqs)
data = data[I]
true_vals = true_vals[I]

num_train_data = int(np.round(0.8*num_seqs))

train_data = data[:num_train_data, :]
train_true_vals = true_vals[:num_train_data]

test_data = data[num_train_data:, :]
test_true_vals = true_vals[num_train_data:]

# Train the network
n = network(seq_length)
n.train(train_data, train_true_vals, test_data, test_true_vals)

# Test its ability to predict function values at higher theta
fig = plt.figure('comparison_fig')
plt.axvline(x=theta[-1], color='r', linestyle='dashed', label='End of training range')

theta = np.append(theta, np.linspace(theta[-1], 1.5*theta[-1], 200))
curve = fun(theta)

fig = plt.figure('comparison_fig')
plt.plot(theta, curve, 'k-', label='Original function')

predictions = np.zeros(theta.size - seq_length)
for i in range(predictions.size):
    predictions[i] = n.output(curve[i:i+seq_length])

plt.plot(theta[seq_length:], predictions, 'b-', label='Predictions')
plt.legend()

plt.show()