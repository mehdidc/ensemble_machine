from load import (load, get_pred, get_scores, split_data,
                  get_all_data, get_test_data, get_models)
import theano
import numpy as np


ramp = 4
obj = load("result_ramp%d.dat" % (ramp,))
pred = get_pred(obj)
scores_csv, scores, contrib = get_scores(obj, "scores_ramp%d.csv" % (ramp,))
num_folds = 1
X_all, y_all, skf, _, _ = get_all_data(num_folds)
X, y = get_test_data(X_all, y_all, skf)
models = get_models(pred, obj["y_dim"])


test_size = 0.25
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
from matplotlib import animation
def logloss(pred, y):
    probs = pred[np.arange(pred.shape[0]), y]
    probs = np.maximum(np.minimum(probs, 1 - 1e-15), 1e-15)
    return -np.mean(np.log(probs))


x_in = layers.InputLayer(shape=(None, X.shape[1]))
h = layers.DenseLayer(x_in, num_units=10,
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
                "accuracy": (self.model.predict(X_train)==y_train).mean(),
                "accuracy_test": (self.model.predict(X_test)==y_test).mean(),
                "logloss_train": (logloss(self.model.predict_proba(X_train), y_train)),
                "logloss_test": (logloss(self.model.predict_proba(X_test), y_test))
        }))

        if (epoch % self.report_each) == 0:
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
                plt.legend()
            plt.show(block=False)
            time.sleep(0.1)
            plt.pause(0.001)
        return res

batch_optimizer = MyBatchOptimizer(max_nb_epochs=100,
                                   optimization_procedure=(updates.adadelta, {"learning_rate" : 1}),
                                   batch_size=200,
                                   patience_nb_epochs=20,
                                   patience_progression_rate_threshold=0.0001,
                                   report_each=5,
                                   verbose=1)

def loss(pred, real, models):
    lambda_ = 0.9
    #loss_kl = (models * T.log(pred.dimshuffle(0, 'x', 1))).mean(axis=(1, 2))
    loss_kl = -T.sqrt(((models - pred.dimshuffle(0, 'x', 1))**2).mean(axis=(1, 2)))
    #loss_kl = models.sum()
    loss =  lambda_ * T.nnet.categorical_crossentropy(pred, real) + (1 - lambda_) * loss_kl
    return loss

model = NeuralNet(nnet_x_to_y, batch_optimizer=batch_optimizer, loss_function=loss)

import matplotlib.pyplot as plt
#plt.ion()
#fig = plt.figure()
#ax = fig.add_subplot(111)
model.fit(X_train, y_train, optional={"models":(models_train)})
plt.show()
