from hp_toolkit.hp import Param, default_eval_functions
import pyneural
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import numpy as np

class PyneuralWrapper(object):
    params = dict(
        num_units = Param(initial=50, interval=[10, 100], type='int'),
        nb_layers = Param(initial=1, interval=[1, 5], type='int'),
        max_iter = Param(initial=10, interval=[10, 50], type='int'),
        batch_size = Param(initial=100, interval=[50, 100, 128, 256, 512], type='choice'),
        learning_rate = Param(initial=0.01, interval=[0, 1], type='real'),
        l2 = Param(initial=0, interval=[0, 0.0001], type='real'),
        decay = Param(initial=1, interval=[0.8, 1], type='real')
    )
    def __init__(self, num_units=50, 
                       nb_layers=1, 
                       max_iter=100, 
                       batch_size=100,
                       learning_rate=0.01,
                       l2=0,
                       decay=1):
        self.num_units = num_units
        self.nb_layers = nb_layers
        self.max_iter =  max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.l2 = l2
        self.decay = decay

        self.model = None
        self.label_encoder = None
        self.label_binarizer = None

    def fit(self, X, y, X_valid=None, y_valid=None, eval_functions=None):
        assert self.model is None
        nb_features = X.shape[1]
        nb_outputs = len(set(tuple(y.tolist())))
        units = [nb_features] + [self.num_units] * self.nb_layers + [nb_outputs]
        self.model = pyneural.NeuralNet(units)
        
        self.label_encoder = LabelEncoder().fit(y)
        y_enc = self.label_encoder.transform(y)
        self.label_binarizer = LabelBinarizer()
        y_bin = self.label_binarizer.fit_transform(y_enc)
        if nb_outputs == 2:
            y_bin = np.concatenate((y_bin, 1 - y_bin), axis=1)
        self.model.train(X, y_bin, self.max_iter, 
                         self.batch_size, 
                         self.learning_rate,
                         self.l2, 
                         self.decay,
                         info=False)

    def predict(self, X):
        assert self.model is not None
        return self.label_encoder.inverse_transform(self.model.predict_label(X))

    def predict_proba(self, X):
        assert self.model is not None
        return self.model.predict_prob(X)
