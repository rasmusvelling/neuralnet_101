import unittest
import numpy as np
import src


class Tests(unittest.TestCase):

    def test_hello(self):
        self.assertEqual("hello world", "hello world")

    def test_initialize_parameters(self):
        # test objects needed
        layer_dimensions = [2, 3, 4, 3, 2]
        X = np.zeros((10, 20))
        nn = src.neuralnet.NN_model()
        params = nn.initialize_parameters(layer_dimensions, X)

        # assert params is a dictionary
        self.assertEqual(type(params), dict)

        # we need number os input/rows from previous layer
        n_lm1 = X.shape[0]

        for l in range(len(layer_dimensions)):
            # get params
            Wl = params['W' + str(l+1)]
            bl = params['b' + str(l+1)]
            n_l = layer_dimensions[l]

            self.assertEqual(Wl.shape, (n_l, n_lm1))
            self.assertEqual(bl.shape, (n_l, 1))

            # set current inputs for next iteration to use
            n_lm1 = n_l

    def test_sigmoid(self):
        nn = src.neuralnet.NN_model()
        a = nn.sigmoid(np.zeros((4, 2)))

        self.assertEqual(a.shape, (4, 2))
        self.assertTrue((
            np.array_equal(
                a,
                np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
            )
        ))

    def test_d_sigmoid(self):
        nn = src.neuralnet.NN_model()
        x = np.zeros((2, 3))
        y = np.array([[.25, .25, .25], [.25, .25, .25]])
        d_sig = nn.d_sigmoid(z=x)

        self.assertEqual(d_sig.shape, (2, 3))
        self.assertTrue(np.allclose(d_sig, y))


    def test_cross_entropy_derivatives(self):
        nn = src.neuralnet.NN_model()
        z = np.random.randn(1, 20) * 0.01
        a = nn.sigmoid(z)
        y = np.random.randint(2, size=(1, 20))

        # this is the short version
        dz1 = a - y

        # this is the long version, but the modular one
        da = nn.d_cross_entropy(y=y, yhat=a)
        dadz = nn.d_sigmoid(z)
        dz2 = da * dadz

        self.assertTrue(np.allclose(dz1, dz2))



if __name__ == '__main__':
    unittest.main()