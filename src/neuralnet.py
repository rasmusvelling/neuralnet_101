import numpy as np

class NN_model(object):

    def __init__(self):
        pass

    def fit(self, X, y, layer_dimensions, iterations, learning_rate):
        # initialize parameters at random
        parameters = self.initialize_parameters(layer_dimensions, X)

        # calculate forward propagation // ie. prediction
        cache = self.forward_propagation(X, parameters, layer_dimensions)

        # calculate loss & cost
        a_L = cache['A' + str(len(layer_dimensions))]
        losses = self.loss_fn(a_L=a_L, y=y)
        cost = self.cost_fn(losses)

        # calculate backwards propagation
        d_a_L = self.d_loss_fn(a_L, y)
        #gradients = self.backwards_propagation()

        

        return d_a_L

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

        n_m1 = X.shape[0]
        for l in range(len(layer_dimensions)):
            n_l = layer_dimensions[l]
            parameters['W' + str(l+1)] = np.random.rand(n_l, n_m1) * .01
            parameters['b' + str(l+1)] = np.zeros((n_l, 1))
            n_m1 = n_l

        # So W1 has dimensions
        return parameters

    def forward_propagation(self, X, parameters, layer_dimensions):

        cache = {}

        a_lm1 = X

        for l in range(len(layer_dimensions)):
            W_l = parameters['W' + str(l+1)]
            b_l = parameters['b' + str(l + 1)]

            z_l = np.dot(W_l, a_lm1) + b_l

            if l == len(layer_dimensions) - 1 :
                a_l = self.sigmoid(z_l)
            else:
                a_l = self.ReLU(z_l)

            a_l = z_l * (z_l > 0)

            cache['A' + str(l + 1)] = a_l
            cache['Z' + str(l + 1)] = z_l

            a_lm1 = a_l

        return cache

    def ReLU(self, Z):
        A = Z * (Z > 0)
        return (A)

    def sigmoid(self, Z):
        A = 1/(1+ np.exp(-Z))
        return A

    def loss_fn(self, a_L, y):
        loss = - y * np.log(a_L) - (1 - y) * np.log(a_L)
        return loss

    def cost_fn(self, loss):
        cost = np.sum(loss)
        return cost

    def d_loss_fn(self, a_L, y):
        d_a_L = -(y/a_L - (1-y)/(1-a_L))
        return d_a_L

    def backwards_propagation(self, layer_dimensions, d_a_L):
        return 0
