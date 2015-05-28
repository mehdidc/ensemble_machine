import matplotlib as mpl
mpl.use('Agg')
import numpy as np
from lasagne.generative.neural_net import NeuralNet
from lasagne.easy import (BatchOptimizer, LightweightModel)
from lasagne import layers, nonlinearities, updates, init
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cross_validation import StratifiedShuffleSplit
import theano.tensor as T
import pandas

def loss_function(pred, real, **params):

    lambda_ = params.get("lambda")
    hidden  = params.get("hidden")
    pre_outputs = params.get("pre_outputs")
    #h_all = T.concatenate([T.shape_padleft(h, 1) for h in hidden], axis=0)
    o_all = T.concatenate([T.shape_padleft(o, 1) for o in pre_outputs], axis=0)

    diff = -lambda_ * T.var(o_all, axis=0).sum()
    return -np.log(pred)[np.arange(y.shape[0]), y].mean() + lambda_ * diff


    return T.nnet.categorical_crossentropy(pred, real)


def logloss(y_pred, y):
    return -np.log(y_pred)[np.arange(y.shape[0]), y].mean()

if __name__ == "__main__":

    X = pandas.read_csv("train.csv")
    y = X["target"]
    X = X.drop(["id", "target"], axis=1)
    X = np.array(X.values)
    y = np.array(y.values)
    y = LabelEncoder().fit_transform(y)
    X = np.log(2 + X)
    X = StandardScaler().fit_transform(X)

    X = X.astype(np.float32)

    X, y = shuffle(X, y)


    class MyBatchOptimizer(BatchOptimizer):

        def iter_update(self, epoch, nb_batches, iter_update_batch):
            status = super(MyBatchOptimizer, self).iter_update(epoch, nb_batches, iter_update_batch)
            status["logloss_train"] = logloss(self.model.predict_proba(self.X_train), self.y_train)
            status["accuracy_train"] = (self.model.predict(self.X_train)==self.y_train).mean()
            status["logloss_valid"] = logloss(self.model.predict_proba(self.X_valid), self.y_valid)
            status["accuracy_valid"] = (self.model.predict(self.X_valid)==self.y_valid).mean()
            return status

    def build_model(nb=40):
        x_in = layers.InputLayer(shape=(None, X.shape[1]))
        params =  dict()
        hidden = []
        pre_outputs = []
        outputs = []
        for i in range(nb):
            h = layers.DenseLayer(x_in, num_units=30,
                                  W=init.GlorotUniform(),
                                  nonlinearity=nonlinearities.rectify)

            hidden.append(h)

            y_pre_i = layers.DenseLayer(h, num_units=len(set(y)),
                                        W=init.GlorotUniform(),
                                        nonlinearity=nonlinearities.linear)
            pre_outputs.append(y_pre_i)
            y_i = layers.NonlinearityLayer(y_pre_i,
                                           nonlinearity=nonlinearities.softmax)
            outputs.append(y_i)

        params["hidden"] = hidden
        params["pre_outputs"] = pre_outputs
        params["outputs"] = outputs

        y_out = layers.ElemwiseSumLayer(outputs, coeffs=1./nb)
        nnet_x_to_y = LightweightModel([x_in],
                                       [y_out])
        return nnet_x_to_y, params

    scores = []
    skf = StratifiedShuffleSplit(y, n_iter=5, test_size=0.1)
    for train_index, test_index in skf:
        model, params = build_model()
        params["lambda"] = 0
        optimizer = MyBatchOptimizer(verbose=1, batch_size=128, max_nb_epochs=50)
        nnet = NeuralNet(model, optimizer, loss_params=params)
        optimizer.X_train = X[train_index]
        optimizer.y_train = y[train_index]
        optimizer.X_valid = X[test_index]
        optimizer.y_valid = y[test_index]
        nnet.fit(X[train_index], y[train_index])

        scores.append(logloss(nnet.predict_proba(X[test_index]), y[test_index]))
    print(scores)
