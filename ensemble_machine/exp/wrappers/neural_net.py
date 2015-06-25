from hp_toolkit.hp import Param, default_eval_functions
from lasagne.generative.neural_net import NeuralNet
from lasagne.easy import (BatchOptimizer, LightweightModel)
from lasagne import layers, nonlinearities, updates, init
from sklearn.preprocessing import LabelEncoder
import theano.tensor as T
from utils import logloss
import numpy as np

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
            return status

class NeuralNetWrapper(object):

    params = dict(
        num_units = Param(initial=50, interval=[10, 500], type='int'),
        nb_layers = Param(initial=1, interval=[1, 3], type='int'),
        batch_size = Param(initial=128, interval=[10, 50, 100, 128, 256, 512], type='choice'),
        learning_rate = Param(initial=1., interval=[-5 * np.log(10), -1 * np.log(10)], type='real', scale='log'),
    )

    def __init__(self, num_units=50, nb_layers=1, batch_size=128, learning_rate=1.):
        self.num_units = int(num_units)
        self.nb_layers = int(nb_layers)
        self.batch_size = int(batch_size)
        self.learning_rate = learning_rate

        self.stats = None
        self.model = None
        self.classes_ = None
        self.n_classes_ = None
        self.label_encoder = None

    def get_params(self, deep=False):
        d = self.__dict__.copy()
        del d["stats"]
        del d["model"]
        del d["classes_"]
        del d["n_classes_"]
        del d["label_encoder"]
        return d
    
    def set_params(self, **p):
        self.__dict__.update(p)

    def fit(self, X, y, sample_weight=None,
            X_valid=None, y_valid=None, eval_functions=default_eval_functions):


        self.label_encoder = LabelEncoder().fit(y)
        # to support AdaBoostClassifier
        self.n_classes_ = len(self.label_encoder.classes_)
        self.classes_ = self.label_encoder.classes_

        y = self.label_encoder.transform(y)
        y = y.astype(np.int32)
        if y_valid is not None:
            y_valid = self.label_encoder.transform(y_valid)
            y_valid = y_valid.astype(np.int32)
        optimizer = MyBatchOptimizer(verbose=1, 
                                     batch_size=self.batch_size, 
                                     optimization_procedure=(updates.rmsprop, {"learning_rate": self.learning_rate}),
                                     max_nb_epochs=100,
                                     patience_nb_epochs=5,
                                     patience_check_each=3,
                                     patience_stat="logloss_valid" if X_valid is not None else "logloss_train",
                                     whole_dataset_in_device=True)
        optimizer.eval_functions = eval_functions
        optimizer.X_train, optimizer.y_train = X, y
        optimizer.X_valid, optimizer.y_valid = X_valid, y_valid
        architecture, params = self.build_architecture(nb_features=X.shape[1], 
                                                       nb_outputs=len(set(tuple(y.tolist()))))
        self.model = NeuralNet(architecture, 
                               optimizer, loss_params=params,
                               loss_function=lambda pred, real, **params: params.get("sample_weight", 1)*T.nnet.categorical_crossentropy(pred, real))
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
