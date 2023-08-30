import numpy as np

class filter:

    def __init__(self, dims):
        self.weights = np.random.randn(dims[0], dims[1])
        self.weights /= self.weights.size

    def apply(self, I):

        out = np.zeros((I.shape[0] + 1 - self.weights.shape[0], I.shape[1] + 1 - self.weights.shape[1])) # Use 'valid' padding.

        for row in range(out.shape[0]):
            for col in range(out.shape[1]):
                out[row, col] = np.sum(self.weights * I[row:row+self.weights.shape[0], col:col+self.weights.shape[1]])

        return out
    
    def backprop(self, I, D):

        for row in range(D.shape[0]):
            for col in range(D.shape[1]):

                delta = D[row, col]
                Isub = I[row:row+self.weights.shape[0], col:col+self.weights.shape[1]]

                self.weights -= delta * Isub
                        
