from sklearn.ensemble import BaggingClassifier
from hp_toolkit.hp import Param, default_eval_functions
from utils import logloss
import numpy as np

class Dummy(object):
    pass

class Bagging(object):

    params = dict(
    )

    def __init__(self, **params):
        self.clf = BaggingClassifier(**params)

    def fit(self, X, y, X_valid=None, y_valid=None, eval_functions=None):
        if eval_functions is None:
            eval_functions = default_eval_functions
        self.clf.fit(X, y)

        self.stats = []

        probas_train = None
        probas_valid = None
        sum_weights = 0.
        for estimator in self.clf.estimators_:
            sum_weights += 1
            t = estimator.predict_proba(X)
            if probas_train is None:
               probas_train = t
            else:
                probas_train += t

            if X_valid is not None:
                v = estimator.predict_proba(X_valid)
                if probas_valid is None:
                    probas_valid = v
                else:
                    probas_valid += v

            stat = {}
            o = Dummy()
            o.predict = lambda X: probas_train.argmax(axis=1)
            o.predict_proba = lambda X: probas_train / sum_weights
            for eval_function_name, eval_function in eval_functions.items():
                val = eval_function(o, X, y)
                stat[eval_function_name + "_train"] = val

            if X_valid is not None and y_valid is not None:
                o = Dummy()
                o.predict = lambda X: probas_valid.argmax(axis=1)
                o.predict_proba = lambda X: probas_valid / sum_weights
                for eval_function_name, eval_function in eval_functions.items():
                    val = eval_function(o, X_valid, y_valid)
                    stat[eval_function_name + "_valid"] = val
            self.stats.append(stat)
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
