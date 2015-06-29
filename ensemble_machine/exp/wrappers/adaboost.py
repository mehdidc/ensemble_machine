from sklearn.ensemble import AdaBoostClassifier
from hp_toolkit.hp import Param, default_eval_functions
from utils import logloss
import numpy as np
from itertools import izip
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LinearRegression
from wrappers.neural_net import NeuralNetWrapper

class Dummy(object):
    pass

class AdaBoost(object):

    params = dict(
            n_estimators=Param(initial=1, interval=[1, 10], type='int'),
    )
    params.update(NeuralNetWrapper.params)

    def __init__(self, **params):
        self.clf = AdaBoostClassifier(n_estimators=params.get("n_estimators"),
                                      base_estimator=NeuralNetWrapper(**params))

    def fit(self, X, y, X_valid=None, y_valid=None, eval_functions=None):
        if eval_functions is None:
            eval_functions = default_eval_functions
        self.clf.fit(X, y)

        self.stats = []

        probas_train = None
        probas_valid = None

        y_train_prob = self.clf.staged_predict_proba(X)
        if X_valid is not None:
            y_valid_prob = self.clf.staged_predict_proba(X_valid)
        else:
            y_valid_prob = [None] * (self.clf.n_estimators)
        k = 0
        for y_train_prob_i, y_valid_prob_i in zip(list(y_train_prob), list(y_valid_prob)):
            #isoto = NeuralNetWrapper()
            #binarizer = LabelBinarizer().fit(y)
            #isoto.fit(y_train_prob_i.astype('float32'), (y.astype('int32')))
            #y_train_prob_i = isoto.predict_proba(y_train_prob_i.astype('float32'))
            stat = {}
            o = Dummy()
            o.predict = lambda X: np.array(y_train_prob_i.argmax(axis=1))
            o.predict_proba = lambda X: np.array(y_train_prob_i)
            for eval_function_name, eval_function in eval_functions.items():
                val = eval_function(o, X, y)
                stat[eval_function_name + "_train"] = val

            if y_valid_prob_i is not None:
                #y_valid_prob_i = isoto.predict_proba(y_valid_prob_i.astype('float32'))
                o = Dummy()
                o.predict = lambda X: np.array(y_valid_prob_i.argmax(axis=1))
                o.predict_proba = lambda X: np.array(y_valid_prob_i)
                for eval_function_name, eval_function in eval_functions.items():
                    val = eval_function(o, X_valid, y_valid)
                    stat[eval_function_name + "_valid"] = val
            self.stats.append(stat)
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

if __name__ == "__main__":
    from datasets import datasets
    from sklearn.ensemble import RandomForestClassifier
    X, y = datasets.get("make_classification")()
    rf = RandomForestClassifier()
    n = AdaBoost(base_estimator=rf, n_estimators=10)
    from hp import find_best_hp, minimize_fn_with_hyperopt
    from sklearn.cross_validation import train_test_split

    X_train, X_valid, y_train, y_valid = train_test_split(X,
                                                          y,
                                                          test_size=0.25)
    n.fit(X_train, y_train, X_valid, y_valid, eval_functions={"logloss": lambda o, X, y: logloss(o.predict_proba(X), y)})
    print(n.stats)
