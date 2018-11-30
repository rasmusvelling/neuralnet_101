import unittest
import numpy as np
import src


class Tests(unittest.TestCase):

    def test_initialize_parameters(self):
        layer_dimensions = [2, 3, 4, 3, 2]
        X = np.zeros((10, 20))
        nn = src.neuralnet.NN_model()

        params = nn.initialize_parameters(layer_dimensions, X)
        self.assertEqual(type(params), dict)

        n_lm1 = X.shape[0]
        for l in range(len(layer_dimensions)):
            Wl = params['W' + str(l+1)]
            bl = params['b' + str(l+1)]
            n_l = layer_dimensions[l]

            self.assertEqual(Wl.shape, (n_l, n_lm1))
            self.assertEqual(bl.shape, (n_l, 1))

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
        d_sig = nn.d_sigmoid(x=np.random.randn(2, 3))
        self.assertEqual(d_sig.shape, (2, 3))

if __name__ == '__main__':
    unittest.main()