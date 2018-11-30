import numpy as np

class NN_model(object):

    def __init__(self):
        pass

    def fit(self, X, y, layer_dimensions, iterations, learning_rate):
        # initialize parameters at random
        parameters = self.initialize_parameters(layer_dimensions, X)
        costs = []
        estimates = []

        for iteration in range(iterations):
            # calculate forward propagation // ie. prediction
            cache = self.forward_propagation(X, parameters, layer_dimensions)

            # calculate loss & cost
            a_L = cache['A' + str(len(layer_dimensions))]
            d_a_L = self.d_loss_fn(a_L, y)
            losses = self.loss_fn(a_L=a_L, y=y)
            cost = self.cost_fn(losses)

            costs.append(cost)
            estimates.append(a_L)


            # calculate backwards propagation

            gradients = self.backwards_propagation(layer_dimensions, parameters, cache, y)
            parameters = self.update_parameters(gradients, parameters, learning_rate)

        return {'costs': costs, 'estimates': estimates}

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
        cache['A0'] = X

        a_lm1 = X

        for l in range(len(layer_dimensions)):
            W_l = parameters['W' + str(l+1)]
            b_l = parameters['b' + str(l + 1)]

            z_l = np.dot(W_l, a_lm1) + b_l

            if l == len(layer_dimensions) - 1 :
                a_l = self.sigmoid(z_l)
            else:
                a_l = self.ReLU(z_l)

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

    def d_sigmoid(self, x):
        ds = self.sigmoid(x) * (1 - self.sigmoid(x))
        return ds

    def loss_fn(self, a_L, y):
        loss = - y * np.log(a_L) - (1 - y) * np.log(a_L)
        return loss

    def cost_fn(self, loss):
        cost = np.sum(loss)
        return cost

    def d_loss_fn(self, a_L, y):
        d_a_L = -(y/a_L - (1-y)/(1-a_L))
        return d_a_L

    def backwards_propagation(self, layer_dimensions, parameters, cache, y):

        L = len(layer_dimensions)
        gradients = {}
        m = y.shape[1]

        for l in reversed(range(L)):

            a_l = cache['A' + str(l + 1)]
            z_l = cache['Z' + str(l + 1)]
            w_l = parameters['W' + str(l + 1)]
            a_lm1 = cache['A' + str(l)]

            if l == L-1:
                d_z_l = a_l - y
            else:
                d_z_l = np.multiply(d_a_l, 1 * (z_l>0))

            d_w_l = (1/m)*(np.dot(d_z_l, a_lm1.T))
            d_b_l = (1/m)*np.sum(d_z_l, axis=1, keepdims=True)
            d_a_lm1 = np.dot(w_l.T, d_z_l)

            gradients['dZ' + str(l + 1)] = d_z_l
            gradients['dW' + str(l + 1)] = d_w_l
            gradients['db' + str(l + 1)] = d_b_l
            gradients['dA' + str(l)] = d_a_lm1

            d_a_l = d_a_lm1

        return gradients

    def update_parameters(self, gradients, parameters, learning_rate):

        for key, value in parameters.items():
            #print(key + " : " + str(parameters[key].shape) + " | " + "d" + key + " : " + str(gradients["d" + key].shape))
            parameters[key] = value - learning_rate * gradients["d" + key]

        return parameters
