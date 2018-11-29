import numpy as np
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

print(data.data.shape)
print(data.target.shape)

class NN(object):

    def __init__(self):
        pass

    def fit(self, X, y, layer_dimensions, iterations, learning_rate):
        # initialize parameters at random
        parameters = self.initialize_parameters(layer_dimensions, X)

        return ""


    def initialize_parameters(self, layer_dimensions, X):
        # X is an (n_x, m) matrix representing n_x features and m samples
        # W1 needs to convert X to a (n_1, m) matrix representing n_1 units in layer 1 and m samples in data
        # Thus W1 needs to be a (n_1, n_x) matrix
        #
        # For W2 we need to convert from (n_1, m) to (n_2, m), thus W2 has dimensions (n_2, n_1).
        # And so on up to L.
        #
        # The b parameters just need to be a vector of n_i

        parameters = {}

        n_m1 = X.shape[1]
        for l in range(len(layer_dimensions)):
            n_l = layer_dimensions[l]
            parameters['W' + str(l+1)] = np.random.randn(n_l, n_m1)
            parameters['b' + str(l+1)] = np.zeros((n_l, 1))
            n_m1 = n_l

        # So W1 has dimensions
        return parameters

nn = NN()
print(nn.initialize_parameters(layer_dimensions=[2,2], X=np.random.randn(3,10)))