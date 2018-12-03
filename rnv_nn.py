import numpy as np
import matplotlib.pyplot as plt
import src
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# load some test data
X, y = make_classification(
    n_features=10, n_redundant=0, n_informative=2,
    random_state=1, n_clusters_per_class=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2, random_state=2)

# settings
layers = [4, 3, 1]
learning_rate = .05
iterations = 5000

layers_sk = tuple(layers[0:(len(layers)-1)])

print(layers_sk)


# My own NN
nn = src.neuralnet.NN_model()
nn_out = nn.fit(X=X_train, y=y_train, layer_dimensions=layers, iterations=iterations, learning_rate=learning_rate)
nn_pred = nn.predict(X=X_test)
nn_acc = (np.sum(nn_pred == y_test)/len(y_test))
print("nn_acc : " + str(nn_acc))


sk_clf = MLPClassifier(
    solver='sgd',
    hidden_layer_sizes=layers_sk,
    learning_rate_init= learning_rate,
    max_iter=iterations,
    random_state=1)
sk_clf.fit(X=X_train, y=y_train)
sk_pred = sk_clf.predict(X_test)
sk_acc = (np.sum(sk_pred == y_test)/len(y_test))
print("sk_acc : " + str(sk_acc))


plt.plot(nn_out['costs'])
plt.show()


print("")






























