import src
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# load some test data
data = load_breast_cancer()


X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target,
    test_size=0.2, random_state=42)

Xt = X_train.T
yt = y_train.reshape((1, len(y_train)))

nn = src.neuralnet.NN_model()
fit_out = nn.fit(X=Xt, y=y_train, layer_dimensions=[2,3,1], iterations=10, learning_rate=.05)
losses = nn.loss_fn(a_L=fit_out, y=yt)
print(fit_out)

































