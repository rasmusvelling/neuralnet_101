import matplotlib.pyplot as plt
import src
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# load some test data
data = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target,
    test_size=0.2, random_state=42)

Xt = X_train.T
yt = y_train.reshape((1, len(y_train)))

# settings
layers = [10,5,5,5,1]
learning_rate=.05
iterations = 200

layers_sk = tuple(layers[0:(len(layers)-1)])

print(layers_sk)


# My own NN
nn = src.neuralnet.NN_model()
nn_out = nn.fit(X=Xt, y=yt, layer_dimensions=[10,5,5,5,1], iterations=iterations, learning_rate=learning_rate)


sk_clf = MLPClassifier(
    solver='sgd',
    hidden_layer_sizes=layers_sk,
    learning_rate_init= learning_rate,
    max_iter=iterations,
    random_state=1)
sk_clf.fit(X=X_train, y=y_train)



plt.plot(fit_out['costs'])
plt.show()

































