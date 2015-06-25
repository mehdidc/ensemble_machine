from hp_toolkit.hp import Param, default_eval_functions
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
    outputs = params.get("outputs")
    #h_all = T.concatenate([T.shape_padleft(h, 1) for h in hidden], axis=0)
    o_all = T.concatenate([T.shape_padleft(o.get_output(params["X_batch"]), 1) for o in outputs], axis=0)
    diff = -T.var(o_all, axis=0).mean(axis=1)

    #loss = 0
    #for o in outputs:
    #    loss += T.nnet.categorical_crossentropy(o.get_output(params["X_batch"]), real)
    #loss += lambda_ * diff
    return T.nnet.categorical_crossentropy(pred, real) + lambda_ * diff
    #return loss

from wrappers.neural_net import MyBatchOptimizer, NeuralNetWrapper

class RepulsiveNeuralNet(object):
    
    params = NeuralNetWrapper.params.copy()
    params.update(dict(
        ensemble_size = Param(initial=5, interval=[1, 10], type='int'),
        lambda_ = Param(initial=0, interval=[0, 5], type='real'),
    ))
    def __init__(self, num_units=50, nb_layers=1, batch_size=128, learning_rate=1., ensemble_size=5, lambda_=0):
        self.num_units = num_units
        self.nb_layers = nb_layers
        self.model = None
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.ensemble_size = ensemble_size
        self.lambda_ = 0

        self.stats = None
        self.model = None
        self.classes_ = None
        self.n_classes_ = None
        self.label_encoder = None

    def fit(self, X, y, X_valid=None, y_valid=None, eval_functions=default_eval_functions):
        optimizer = MyBatchOptimizer(verbose=2, 
                                     batch_size=self.batch_size, 
                                     max_nb_epochs=100, 
                                     patience_nb_epochs=5,
                                     patience_stat="logloss_valid" if X_valid is not None else "logloss_train",
                                     whole_dataset_in_device=True)
        optimizer.X_train, optimizer.y_train = X, y
        optimizer.X_valid, optimizer.y_valid = X_valid, y_valid
        optimizer.eval_functions = eval_functions
        architecture, params = self.build_architecture(nb_features=X.shape[1], 
                                                       nb_outputs=len(set(tuple(y.tolist()))))
        params["lambda"] = self.lambda_
        self.model = NeuralNet(architecture, optimizer, 
                               loss_params=params,
                               loss_function=loss_function)
        self.model.fit(X, y)
        self.stats = optimizer.stats

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def build_architecture(self, nb_features, nb_outputs):
        x_in = layers.InputLayer(shape=(None, nb_features))
        params =  dict()
        hidden = []
        pre_outputs = []
        outputs = []
        collab = []
        for i in range(self.ensemble_size):
            h = layers.DenseLayer(x_in, num_units=self.num_units,
                                  W=init.GlorotUniform(),
                                  nonlinearity=nonlinearities.rectify)
            hidden.append(h)
            y_pre_i = layers.DenseLayer(h, num_units=nb_outputs,
                                        W=init.GlorotUniform(),
                                        nonlinearity=nonlinearities.linear)

            pre_outputs.append(y_pre_i)
            y_i = layers.NonlinearityLayer(y_pre_i,
                                           nonlinearity=nonlinearities.softmax)
            outputs.append(y_i)

        params["hidden"] = hidden
        params["pre_outputs"] = pre_outputs
        params["outputs"] = outputs

        y_out = layers.ElemwiseSumLayer(outputs, coeffs=1./self.ensemble_size)
        nnet_x_to_y = LightweightModel([x_in],
                                       [y_out])
        return nnet_x_to_y, params

if __name__ == "__main__":
    from datasets import datasets
    X, y = datasets.get("make_classification")()
    n = RepulsiveNeuralNet(lambda_=0.1)
    from hp import find_best_hp, minimize_fn_with_hyperopt
    from sklearn.cross_validation import train_test_split

    X_train, X_valid, y_train, y_valid = train_test_split(X, 
                                                          y, 
                                                          test_size=0.25)
    best_hp, _ = find_best_hp(RepulsiveNeuralNet,
                              minimize_fn_with_hyperopt,
                              X_train,
                              X_valid,
                              y_train,
                              y_valid,
                              max_evaluations=10)
    print(best_hp)
