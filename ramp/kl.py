from load import (load, get_pred, get_scores, split_data,
                  get_all_data, get_test_data, get_models)
import theano
import numpy as np

import os

from lightexperiments.light import Light

light = Light()
light.launch()
light.initials()
light.file_snapshot()

seed = 151231
np.random.seed(seed)
light.set_seed(seed)

light.tag("mimic_probabilities")


ramp_dir = os.path.join(os.getenv("DATA_PATH"), "ramp")
ramp = 4
num_prediction = 100
obj = load(os.path.join(ramp_dir, "result_ramp%d.dat" % (ramp,)))

class FeatureExtractor(object):

    def __init__(self):
        pass

    def fit(self, X_dict, y):
        pass

    def transform(self, X_dict):
        cols = X_dict[0].keys()
        x =  np.array([[instance[col] for col in cols] for instance in X_dict])
        return np.log(2. + x)


pred = get_pred(obj, num_prediction=num_prediction)
scores_csv, scores, contrib = get_scores(obj, os.path.join(ramp_dir, "scores_ramp%d.csv" % (ramp,)))
num_folds = 1
X_all, y_all, skf, _, _ = get_all_data(num_folds, FeatureExtractor=FeatureExtractor)



X, y = get_test_data(X_all, y_all, skf, num_prediction=num_prediction)
models = get_models(pred, obj["y_dim"])


test_size = 0.1
to = int(X.shape[0]*(1-test_size))
X_train, y_train = X[0:to], y[0:to]
X_test, y_test = X[to:], y[to:]
y_dim = obj["y_dim"]
#models = np.concatenate((o[2), axis=0)
models = models[:, 1:2, :]
models_train, models_test = models[0:to], models[to:]

from lasagne import init, nonlinearities, layers, updates
from lasagne.generative.neural_net import NeuralNet
from lasagne.easy import BatchOptimizer, LightweightModel, get_stat
import theano.tensor as T
from collections import OrderedDict
def logloss(pred, y):
    probs = pred[np.arange(pred.shape[0]), y]
    probs = np.maximum(np.minimum(probs, 1 - 1e-15), 1e-15)
    return -np.mean(np.log(probs))

num_units = 1000
light.set("num_units", num_units)
x_in = layers.InputLayer(shape=(None, X.shape[1]))
h = layers.DenseLayer(x_in, num_units=num_units,
                      W=init.GlorotUniform(),
                      nonlinearity=nonlinearities.rectify)
y_out = layers.DenseLayer(h, num_units=9,
                          W=init.GlorotUniform(),
                          nonlinearity=nonlinearities.softmax)
nnet_x_to_y = LightweightModel([x_in],
                               [y_out])

import time

class MyBatchOptimizer(BatchOptimizer):

    def iter_update(self, epoch, nb_batches, iter_update_batch):
        res = super(MyBatchOptimizer, self).iter_update(epoch, nb_batches, iter_update_batch)
        res.update(OrderedDict({
                "accuracy_train": (self.model.predict(X_train)==y_train).mean(),
                "accuracy_test": (self.model.predict(X_test)==y_test).mean(),
                "logloss_train": (logloss(self.model.predict_proba(X_train), y_train)),
                "logloss_test": (logloss(self.model.predict_proba(X_test), y_test))
        }))
        
        for k, v in res.items():
            light.append(k, v)

        if (epoch % self.report_each) == 0:
            plt.subplot(1, 2, 1)
            plt.plot(get_stat("epoch", self.stats),
                    get_stat("logloss_train", batch_optimizer.stats),
                    c='blue', label='logloss_train')
            plt.plot(get_stat("epoch", self.stats),
                    get_stat("logloss_test", batch_optimizer.stats),
                    c='green', label='logloss_test')
            plt.plot(get_stat("epoch", self.stats),
                     [logloss(models_train[:, 0, :], y_train)] * len(self.stats),
                     c='red', label='logloss_delf_train'
                     )
            plt.plot(get_stat("epoch", self.stats),
                     [logloss(models_test[:, 0, :], y_test)] * len(self.stats),
                     c='black', label='logloss_delf_test'
                     )
            if len(self.stats)==1:
                plt.legend(loc='upper right')

            plt.subplot(1, 2, 2)

            plt.plot(get_stat("epoch", self.stats),
                     get_stat("accuracy_train", self.stats),
                     c='blue', label='accuracy_train'
                     )
            plt.plot(get_stat("epoch", self.stats),
                     get_stat("accuracy_test", self.stats),
                     c='green', label='accuracy_test'
                     )
            plt.plot(get_stat("epoch", self.stats),
                    [(models_train[:, 0, :].argmax(axis=1)==y_train).mean()] * len(self.stats),
                     c='red', label='accuracy_delf_train'
                     )
            plt.plot(get_stat("epoch", self.stats),
                     [(models_test[:, 0, :].argmax(axis=1)==y_test).mean()] * len(self.stats),
                     c='black', label='accuracy_delf_test'
                     )
            if len(self.stats)==1:
                plt.legend(loc='lower left')
            plt.show(block=False)
            time.sleep(0.1)
            plt.pause(0.001)
        return res

batch_optimizer = MyBatchOptimizer(max_nb_epochs=2000,
        optimization_procedure=(updates.sgd, {"learning_rate" : 0.001}),
                                   batch_size=10,
                                   patience_nb_epochs=20,
                                   patience_progression_rate_threshold=1e-5,
                                   report_each=5,
                                   verbose=1)

def loss(pred, real, models):
    lambda_ = 0.
    #loss_kl = (models * T.log(pred.dimshuffle(0, 'x', 1))).mean(axis=(1, 2))
    loss_kl = (((models - pred.dimshuffle(0, 'x', 1))**2).mean(axis=(1, 2)))
    #loss_kl = models.sum()
    loss =  lambda_ * T.nnet.categorical_crossentropy(pred, real) + (1 - lambda_) * loss_kl
    return loss

model = NeuralNet(nnet_x_to_y, batch_optimizer=batch_optimizer, loss_function=loss)

import matplotlib.pyplot as plt
model.fit(X_train, y_train, optional={"models":(models_train)})
light.endings()
light.store_experiment()
light.close()
plt.show()
