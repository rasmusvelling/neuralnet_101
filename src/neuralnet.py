from copy import deepcopy
import numpy as np

class NN_model(object):


    # We could pass model parameters at the init stage. Maybe I'll change that later.
    def __init__(self):
        pass


    def fit(self, X, y, layer_dimensions, iterations, learning_rate):
        """Trains model and generates parameters (weights) based on input data X,y and hyper parameters."""

        # prepare X and y
        Xt = self._reshape_X(X)
        print("Xt.shape : " + str(Xt.shape))
        yt = self._reshape_y(y)
        print("yt.shape : " + str(yt.shape))

        # initialize parameters at random
        parameters = self.initialize_parameters(layer_dimensions, Xt)

        # lists for storing output
        costs = []
        estimates = []

        # update weights along iterations.
        # TODO: add tolerance parameter to auto-stop iterations when tolerance level is met.
        for iteration in range(iterations):

            # calculate forward propagation // ie. prediction
            cache = self.forward_propagation(Xt, parameters, layer_dimensions)

            # get yhat - the latest activation layer
            a_L = cache['A' + str(len(layer_dimensions))]

            # calculate backwards propagation
            gradients = self.backwards_propagation(layer_dimensions, parameters, cache, yt)

            # update parameters
            parameters = self.update_parameters(gradients, parameters, learning_rate)

            if iteration % 500 == 0:
                cost = self.cost_cross_entropy(y=yt, yhat=a_L)
                costs.append(cost)
                estimates.append(a_L)

                print("Cost after iteration %i: %f" % (iteration, cost))

        # all-iterations have been completed, time to save some results and return something
        self.parameters = parameters
        self.layer_dimensions = layer_dimensions

        return {'costs': costs, 'estimates': estimates}


    def initialize_parameters(self, layer_dimensions, X):
        """
        Initializes model parameters (weights) at random.

        A note about dimensions
        X is an (n_x, m) matrix representing n_x features and m samples
        W1 needs to convert X to a (n_1, m) matrix representing n_1 units in layer 1 and m samples in data
        Thus W1 needs to be a (n_1, n_x) matrix

        For W2 we need to convert from (n_1, m) to (n_2, m), thus W2 has dimensions (n_2, n_1).
        And so on up to L.

        The b parameters just need to be a vector of n_i
        """

        parameters = {}

        n_m1 = X.shape[0]
        for l in range(len(layer_dimensions)):
            n_l = layer_dimensions[l]
            parameters['W' + str(l+1)] = np.random.randn(n_l, n_m1) * .1
            parameters['b' + str(l+1)] = np.zeros((n_l, 1))
            n_m1 = n_l

        # So W1 has dimensions
        return parameters

    ########################
    # Forwards and backwards propagations
    #

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


    def backwards_propagation(self, layer_dimensions, parameters, cache, y):
        """ Backwards propagation calculates the derivative of the loss function with respect to all parameters
        (weights). These derivate are then used to update the parameters. """

        # L is the number of layer, we need this for iteration backwards
        L = len(layer_dimensions)
        # Object for storing gradients of the weights down the layers
        gradients = {}
        # number of sampleas in data
        m = y.shape[1]

        # iterate backwards over layers
        for l in reversed(range(L)):

            # cached object to calculate gradients
            a_l = cache['A' + str(l + 1)]
            z_l = cache['Z' + str(l + 1)]
            w_l = parameters['W' + str(l + 1)]
            a_lm1 = cache['A' + str(l)]

            #if on top layer // calculating dz
            if l == L-1:
                d_z_l = self.d_cross_entropy(y=y, yhat=a_l) * self.d_sigmoid(z=z_l)
            else:
                d_z_l = np.multiply(d_a_l, self.d_relu(z_l))

            # other gradients
            d_w_l = (1/m)*(np.dot(d_z_l, a_lm1.T))
            d_b_l = (1/m)*np.sum(d_z_l, axis=1, keepdims=True)
            d_a_lm1 = np.dot(w_l.T, d_z_l)

            gradients['dZ' + str(l + 1)] = d_z_l
            gradients['dW' + str(l + 1)] = d_w_l
            gradients['db' + str(l + 1)] = d_b_l
            gradients['dA' + str(l)] = d_a_lm1

            # pass gradient of a in the layer below to next iteration
            d_a_l = d_a_lm1

        return gradients


    ########################
    # Activation functions
    #

    def ReLU(self, Z):
        A = Z * (Z > 0)
        return (A)

    def d_relu(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def sigmoid(self, Z):
        A = 1/(1+ np.exp(-Z))
        return A

    def d_sigmoid(self, z):
        ds = self.sigmoid(z) * (1 - self.sigmoid(z))
        return ds


    ########################
    # Loss functions
    #

    def cost_cross_entropy(self, y, yhat):
        loss = y * np.log(yhat) + (1-y) * np.log(1-yhat)
        cost = -np.sum(loss) / loss.shape[1]
        return cost

    def d_cross_entropy(self, y, yhat):
        d_cc = - y / yhat + (1-y)/(1-yhat)
        return d_cc



    ########################
    # Other functions
    #

    def update_parameters(self, gradients, parameters, learning_rate):
        for key, value in parameters.items():
            parameters[key] = value - learning_rate * gradients["d" + key]
        return parameters


    def predict(self, X, output_type="flatten"):
        """ Predicts yhat given a dataset X"""

        # Reshape X (transpose)
        X = self._reshape_X(X)

        # Get parameters
        parameters = self.parameters
        # Get number of layers
        layer_dimensions = self.layer_dimensions

        # make forward prop using paramters estimated in fit
        cache = self.forward_propagation(X, parameters, layer_dimensions)
        # get output of last layer
        a_L = cache['A' + str(len(layer_dimensions))]

        # since we're doing classification, assign value based on probability
        yhat = 1 * (a_L > .5)

        # transpose yhat before outputting
        yhat = yhat.T

        if output_type == "flatten":
            yhat = yhat.flatten()
        elif output_type == "list":
            yhat = yhat.flatten().tolist()

        return yhat


    def _reshape_y(self, y):
        # reshape y to row vector
        y = deepcopy(y)
        if type(y) == list:
            m = len(y)
            y = np.array(y)
            y = y.reshape((1, m))
        elif y.ndim == 1:
            m = y.shape[0]
            y = y.reshape((1, m))
        elif y.ndim == 2:
            m = y.shape[0] * y.shape[1]
            y = y.reshape((1, m))
        return y

    def _reshape_X(self, X):
        # Transpose X
        X = deepcopy(X)
        X = X.T
        return X
