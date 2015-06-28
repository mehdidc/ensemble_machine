from hp_toolkit.hp import Param, default_eval_functions
from lasagne.generative.neural_net import NeuralNet
from lasagne.easy import (BatchOptimizer, LightweightModel)
from lasagne import layers, nonlinearities, updates, init
from sklearn.preprocessing import LabelEncoder
import theano.tensor as T
from utils import logloss
import numpy as np

import theano

class MyBatchOptimizer(BatchOptimizer):
        def iter_update(self, epoch, nb_batches, iter_update_batch):
            #ss = range(int(self.X_train.shape[0] * 0.3))
            ss = range(self.X_train.shape[0])
            status = super(MyBatchOptimizer, self).iter_update(epoch, nb_batches, iter_update_batch)
            status["logloss_train"] = float(logloss(self.model.predict_proba(self.X_train[ss]), self.y_train[ss]))
            if self.X_valid is not None and self.y_valid is not None:
                status["loss_valid"] = float(self.model.get_loss(self.X_valid, self.y_valid))
            status["accuracy_train"] = float((self.model.predict(self.X_train[ss])==self.y_train[ss]).mean())
            if self.X_valid is not None and self.y_valid is not None:
                status["logloss_valid"] = float(logloss(self.model.predict_proba(self.X_valid), self.y_valid))
                status["accuracy_valid"] = float((self.model.predict(self.X_valid)==self.y_valid).mean())

            for name, func in self.eval_functions.items():
                status["{0}_train".format(name)] = float(func(self.model, self.X_train[ss], self.y_train[ss]))
                if self.X_valid is not None and self.y_valid is not None:
                    status["{0}_valid".format(name)] = float(func(self.model, self.X_valid, self.y_valid))
            for k, v in status.items():
                status[k] = float(v)

            # update learning rate
            factor = np.array(1. - self.learning_rate_annealing, dtype='float32')
            self.cur_learning_rate.set_value(self.cur_learning_rate.get_value() * factor)
            return status

class NeuralNetWrapper(object):

    params = dict(
        num_units = Param(initial=50, interval=[10, 500], type='int'),
        nb_layers = Param(initial=1, interval=[1, 3], type='int'),
        batch_size = Param(initial=128, interval=[10, 50, 100, 128, 256, 512], type='choice'),
        learning_rate = Param(initial=1., interval=[-5, -1], type='real', scale='log10'),
        learning_rate_annealing = Param(initial=0., interval=[-10, -5], scale='log10'),
    )

    def __init__(self, num_units=50, nb_layers=1, batch_size=128,
                 learning_rate=1.,
                 learning_rate_annealing=0.,
                 max_nb_epochs=100,
                 early_stopping_on=None):
        self.num_units = int(num_units)
        self.nb_layers = int(nb_layers)
        self.batch_size = int(batch_size)
        self.learning_rate = learning_rate
        self.learning_rate_annealing = learning_rate_annealing
        self.max_nb_epochs = int(max_nb_epochs)
        self.early_stopping_on = early_stopping_on

        self.stats = None
        self.model = None
        self.classes_ = None
        self.n_classes_ = None
        self.label_encoder = None
        self.cur_learning_rate = None

    def get_params(self, deep=False):
        d = self.__dict__.copy()
        del d["stats"]
        del d["model"]
        del d["classes_"]
        del d["n_classes_"]
        del d["label_encoder"]
        del d["cur_learning_rate"]
        return d

    def set_params(self, **p):
        self.__dict__.update(p)

    def fit(self, X, y, sample_weight=None,
            X_valid=None, y_valid=None,
            eval_functions=default_eval_functions):


        self.label_encoder = LabelEncoder().fit(y)
        # to support AdaBoostClassifier
        self.n_classes_ = len(self.label_encoder.classes_)
        self.classes_ = self.label_encoder.classes_

        y = self.label_encoder.transform(y)
        y = y.astype(np.int32)
        if y_valid is not None:
            y_valid = self.label_encoder.transform(y_valid)
            y_valid = y_valid.astype(np.int32)

        cur_learning_rate = theano.shared(np.array(self.learning_rate, dtype="float32"))
        params = dict(verbose=1,
                      batch_size=self.batch_size,
                      optimization_procedure=(updates.rmsprop, {"learning_rate": cur_learning_rate}),
                      max_nb_epochs=self.max_nb_epochs,
                      whole_dataset_in_device=True)
        if self.early_stopping_on is not None:
            params.update(dict(patience_nb_epochs=5,
                               patience_check_each=3,
                               patience_stat="logloss_{0}".format(self.early_stopping_on)))
        optimizer = MyBatchOptimizer(**params)
        optimizer.eval_functions = eval_functions
        optimizer.X_train, optimizer.y_train = X, y
        optimizer.X_valid, optimizer.y_valid = X_valid, y_valid
        optimizer.cur_learning_rate = cur_learning_rate
        optimizer.learning_rate_annealing = self.learning_rate_annealing
        architecture, params = self.build_architecture(nb_features=X.shape[1],
                                                       nb_outputs=len(set(tuple(y.tolist()))))
        self.model = NeuralNet(architecture,
                               optimizer, loss_params=params,
                               loss_function=lambda pred, real, **params: (params.get("sample_weight", 1)/params.get("sample_weight", np.array(1, dtype='float32')).sum())*T.nnet.categorical_crossentropy(pred, real))
        optional = dict()
        if sample_weight is not None:
            optional["sample_weight"] = sample_weight.astype(np.float32)
        self.model.fit(X, y, optional=optional)
        self.stats = optimizer.stats

    def predict(self, X):
        return self.label_encoder.inverse_transform(self.model.predict(X))

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def build_architecture(self, nb_features, nb_outputs):
        x_in = layers.InputLayer(shape=(None, nb_features))
        params =  dict()
        h = x_in
        for i in range(self.nb_layers):
            h = layers.DenseLayer(h, num_units=self.num_units,
                                  W=init.GlorotUniform(),
                                  nonlinearity=nonlinearities.rectify)
        y_out = layers.DenseLayer(h, num_units=nb_outputs,
                                     nonlinearity=nonlinearities.softmax)
        nnet_x_to_y = LightweightModel([x_in],
                                       [y_out])
        return nnet_x_to_y, params
