import mds
import theano
import theano.tensor as T
from lasagne import easy, updates, layers, init, nonlinearities
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
from sklearn.decomposition import KernelPCA, PCA, RandomizedPCA, TruncatedSVD


from lasagne.easy import LightweightModel


def ensemble_machine_model_loss(y_new, z_new,
                                y_old, z_old,
                                dist_y_func, dist_z_func):
    # y_new : (nb new models, nb outputs) predicted points by the new models
    # z_new : (nb new models, embedding dim) wanted points in the
    #         embedding
    # y_old : (nb models, nb outputs)
    # z_old : (nb models, embedding dim)

    y = T.concatenate([y_old, y_new], axis=0)
    dist_y = dist_y_func(y)[y_old.shape[0]:]

    z = T.concatenate([z_old, z_new], axis=0)
    dist_z = dist_z_func(z)[z_old.shape[0]:]

    return mds.MDS_loss(dist_y, dist_z)


def ensemble_machine_model_loss_given_y(y, wanted_y):
    return T.sqrt((((y - wanted_y) ** 2).sum(axis=1))).mean()


def model_representation_by_increasing_mean(probas, context=T):
    return (probas * context.arange(probas.shape[1])).sum(axis=1)


def model_representation_by_labels(probas, context=T):
    return probas.argmax(axis=1)


def model_representation_by_probas(probas, context=T):
    return probas.flatten()


#def euclidean(X):
#    d = (X.reshape( (X.shape[0], X.shape[1], 1) ) -
#         X.T.reshape( (1, X.shape[1], X.shape[0])  )) ** 2
#    return np.sqrt(d.sum(axis=1))

from sklearn.metrics.pairwise import euclidean_distances
def euclidean(X):
    return euclidean_distances(X)


class EnsembleMachine(object):

    def __init__(self, n_components=2,
                 dist_y=mds.euclidian_dist,
                 dist_z=mds.euclidian_dist,
                 batch_optimizer=None):

        self.n_components = n_components
        if batch_optimizer is None:
            batch_optimizer = easy.BatchOptimizer()
        self.batch_optimizer = batch_optimizer
        self.dist_y = dist_y
        self.dist_z = dist_z

        self.Z = None

        self.inverser_ = None

    def fit_with_pca(self, Y):
        assert (self.dist_y == mds.euclidian_dist and
                self.dist_z == mds.euclidian_dist)
        self.inverser_ = KernelPCA(n_components=2, fit_inverse_transform=True,
                                   kernel='linear')

        Z = self.inverser_.fit_transform(Y)

        d_y = euclidean(Y)
        d_z = euclidean(Z)
        self.stress_ = (((d_y - d_z) ** 2).sum())

        self.Y = theano.shared(Y, borrow=True)
        self.Z = theano.shared(Z, borrow=True)

    def fit_with_mds(self, Y, **par):
        assert self.dist_z == mds.euclidian_dist

        params = dict(n_components=2, n_init=10, verbose=1)
        params.update(par)
        if self.dist_y == mds.hamming_dist:
            mds_ = MDS(dissimilarity='precomputed', **params)
            dis = ((Y[:, :, np.newaxis] !=
                    Y.T[np.newaxis, :, :]).sum(axis=1))
            d_y = dis
            Z = mds_.fit_transform(dis)
        elif self.dist_y == mds.euclidian_dist:
            mds_ = MDS(dissimilarity='euclidean', **params)
            d_y = euclidean(Y)
            Z = mds_.fit_transform(Y)
        d_z = euclidean(Z)
        self.stress_ = (((d_y - d_z) ** 2).sum())
        self.Y = theano.shared(Y, borrow=True)
        self.Z = theano.shared(Z, borrow=True)
        return self

    def fit_with_gradient_descent(self, Y):

        nb_examples = Y.shape[0]

        Z_init = np.random.uniform(size=(Y.shape[0], self.n_components))
        Z_init = Z_init.astype(theano.config.floatX)
        self.Z = theano.shared(Z_init, borrow=True)

        Y = theano.shared(Y, borrow=True)
        Y_batch = T.matrix('Y_batch')
        batch_index = T.iscalar('batch_index')
        batch_slice = easy.get_batch_slice(batch_index,
                                           self.batch_optimizer.batch_size)
        Z_batch = self.Z[batch_slice]

        d_y = self.dist_y(Y_batch)
        d_z = self.dist_z(Z_batch)

        loss = mds.MDS_loss(d_y, d_z)

        all_params = [self.Z]

        opti_function, opti_kwargs = self.batch_optimizer.optimization_procedure
        updates = opti_function(loss, all_params, **opti_kwargs)

        givens = {
            Y_batch: Y[batch_slice],
            Z_batch: self.Z[batch_slice]
        }
        iter_train = theano.function(
            [batch_index], loss,
            updates=updates,
            givens=givens
        )
        nb_batches = easy.get_nb_batches(nb_examples,
                                         self.batch_optimizer.batch_size)
        self.batch_optimizer.optimize(nb_batches, iter_train)


        get_d_Y = theano.function([], d_y, givens={Y_batch: Y})
        get_d_Z = theano.function([], d_z, givens={Z_batch: self.Z})

        self.stress_ = ((get_d_Y() - get_d_Z()) ** 2).sum()
        self.Y = Y
        return self

    def update_with_gradient_descent(self, X, y, models_new, Z_new,
                                     batch_optimizer=None, lambda_=1.,
                                     model_repr=model_representation_by_probas,
                                     inverser=False):
        assert X.shape[0] == y.shape[0]
        assert len(models_new) == Z_new.shape[0]

        nb_examples = X.shape[0]
        if batch_optimizer is None:
            batch_optimizer = easy.BatchOptimizer(batch_size=X.shape[0],
                                                  verbose=1)

        X_batch = T.matrix('X_batch')
        y_batch = T.ivector('y_batch')
        batch_index = T.iscalar('batch_index')
        batch_slice = easy.get_batch_slice(batch_index,
                                           batch_optimizer.batch_size)

        # X : dataset (nb examples, nb features)
        # Z_new (nb_new_models, nb_components)
        # models_new : list of nb_new_models LightweightModel
        outputs = []
        pred = []
        probas = []
        for model in models_new:
            output, = model.get_output(X_batch)
            pred.append(output.argmax(axis=1))
            probas.append(output)
            output = model_repr(output)
            output = output.dimshuffle('x', 0)
            outputs.append(output)  # (1, nb examples)

        # (nb_new_models, nb_examples)
        Y_new = T.concatenate(outputs, axis=0)

        if inverser is False:
            loss_ensemble_machine = ensemble_machine_model_loss(Y_new, Z_new,
                                                                self.Y, self.Z,
                                                                self.dist_y,
                                                                self.dist_z)
        else:
            assert self.inverser_ is not None
            Y_wanted = self.inverser_.inverse_transform(Z_new)
            loss_ensemble_machine = ensemble_machine_model_loss_given_y(Y_new,
                                                                        Y_wanted)
        C = 1.
        if model_repr == model_representation_by_probas:
            if inverser is True:
                y_dim = probas[0].shape[1]
                C = 1. / y_dim
                loss_ensemble_machine *= C # make it comparable with loss_accuracy
            else:
                C = 1. / self.Y.get_value().shape[0]
                loss_ensemble_machine *= C # make it comparable with loss_accuracy

        loss_accuracy = 0.
        for pr in probas:
            loss_accuracy += -T.mean(T.log(pr)[T.arange(y_batch.shape[0]),
                                               y_batch])


        loss = lambda_ * loss_accuracy + (1 - lambda_) * loss_ensemble_machine
        all_params = list(set(param
                              for model in models_new
                              for param in model.get_all_params()))

        opti_function, opti_kwargs = batch_optimizer.optimization_procedure
        updates = opti_function(loss, all_params, **opti_kwargs)

        X = theano.shared(X, borrow=True)
        y = theano.shared(y, borrow=True)
        givens = {
            X_batch: X[batch_slice],
            y_batch: y[batch_slice]
        }
        iter_train = theano.function(
            [batch_index], loss,
            updates=updates,
            givens=givens
        )

        for i, model in enumerate(models_new):
            pr = probas[i]
            predict = theano.function([X_batch], pred[i])
            predict_proba = theano.function([X_batch], pr)
            self.__dict__["predict_%d" % (i,)] = predict
            self.__dict__["predict_proba_%d" % (i,)] = predict_proba
        self.predict = self.predict_0
        self.predict_proba = self.predict_proba_0

        self.get_loss = theano.function([X_batch, y_batch], loss)
        self.get_loss_ensemble_machine = theano.function([X_batch],
                                                         loss_ensemble_machine)
        self.get_loss_accuracy = theano.function([X_batch, y_batch],
                                                 loss_accuracy)

        y = T.concatenate([self.Y, Y_new], axis=0)
        dist_y = self.dist_y(y)

        z = T.concatenate([self.Z, Z_new], axis=0)
        dist_z = self.dist_z(z)

        self.get_dist_y = theano.function([X_batch], dist_y)
        self.get_dist_z = theano.function([], dist_z)

        nb_batches = easy.get_nb_batches(nb_examples,
                                         batch_optimizer.batch_size)
        batch_optimizer.optimize(nb_batches, iter_train)
        return self


def build_Y(X, models, n_classes, model_repr=model_representation_by_probas):
    name, model = models[0]
    r = model_repr(model.predict_proba(X), context=np)
    Y = np.zeros((len(models), r.shape[0]))
    Y[0] = r
    for i, (name, model) in enumerate(models[1:]):
        Y[i + 1, :] = model_repr(model.predict_proba(X), context=np)
    return Y

if __name__ == "__main__":
    from lightexperiments.light import Light

    light = Light()
    from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                                  GradientBoostingClassifier)
#    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler
#   import matplotlib.pyplot as plt
    from plotting import plot_model_embeddings

    light.launch()

    light.initials()
    light.file_snapshot()
    light.tag("ensemble_model_example")

    seed = 1
    np.random.seed(seed)
    light.set_seed(seed)

    y_dim = 10
    n_classes = y_dim
    params = dict(n_classes=y_dim, n_informative=10)
    X, y = make_classification(200, **params)
    X = StandardScaler().fit_transform(X)
    X = X.astype(theano.config.floatX)
    y = y.astype('int32')

    models = [
        ('rf_1', RandomForestClassifier(n_estimators=50, max_depth=5)),
        ('rf_2', RandomForestClassifier(n_estimators=80, max_depth=10)),
        ('rf_3', RandomForestClassifier(n_estimators=80, max_depth=3)),
        ('ada_1', AdaBoostClassifier(n_estimators=10)),
        ('svm_1', SVC(probability=True)),
#       ('gb_1', GradientBoostingClassifier(max_depth=3)),
#       ('gb_2', GradientBoostingClassifier(max_depth=8)),
    ]
    for _, model in models:
        model.fit(X, y)
    Y = build_Y(X, models, n_classes)
    Y = Y.astype(theano.config.floatX)
    opti = (updates.sgd, {"learning_rate": 0.1})
    batch_optimizer = easy.BatchOptimizer(verbose=1,
                                          batch_size=Y.shape[0],
                                          max_nb_epochs=10,
                                          optimization_procedure=opti)
    es = EnsembleMachine(n_components=2,
                         dist_y=mds.euclidian_dist,
                         batch_optimizer=batch_optimizer)
    es.fit_with_mds(Y)

    plot_model_embeddings(models, es.Z.get_value(), save_file="before.png")


    x_in = layers.InputLayer(shape=(None, X.shape[1]))
    h = layers.DenseLayer(x_in, num_units=200,
                          W=init.GlorotNormal(),
                          nonlinearity=nonlinearities.rectify)
    h = layers.DenseLayer(x_in, num_units=200,
                          W=init.GlorotNormal(),
                          nonlinearity=nonlinearities.rectify)
    z_out = layers.DenseLayer(h, num_units=y_dim,
                              W=init.GlorotUniform(),
                              nonlinearity=nonlinearities.softmax)
    model = LightweightModel([x_in],
                             [z_out])
    models_new = [
        model
    ]
    Z_new = [
        es.Z.get_value()[0].tolist()
    ]
    Z_new = np.array(Z_new, dtype=theano.config.floatX)

    class MyBatchOptimizer(easy.BatchOptimizer):
        def iter_update(self, epoch, nb_batches,
                        iter_update_batch):
                super(MyBatchOptimizer, self).iter_update(epoch,
                                                          nb_batches,
                                                          iter_update_batch)
                if epoch == self.max_nb_epochs - 1:
                    print(es.get_dist_y(X))
                    print(es.get_dist_z())

                loss_ensemble_machine = float(es.get_loss_ensemble_machine(X))
                loss_accuracy = float(es.get_loss_accuracy(X, y))
                loss = float(es.get_loss(X, y))
                light.append("loss_ensemble_machine", loss_ensemble_machine)
                light.append("loss_accuracy", loss_accuracy)
                light.append("loss", loss)

                print("loss ensemble machine : ", loss_ensemble_machine)
                print("loss accuracy : ", loss_accuracy)
                print("loss :", loss)

    procedure = (updates.nesterov_momentum,
                 {"learning_rate": 0.001, "momentum": 0.8})
    batch_optimizer = MyBatchOptimizer(verbose=1,
                                       max_nb_epochs=100,
                                       batch_size=X.shape[0],
                                       optimization_procedure=procedure)
    print(X.shape)
    print(y.shape)
    print(len(models_new))
    print(Z_new)
    es.update_with_gradient_descent(X, y, models_new, Z_new,
                                    batch_optimizer=batch_optimizer,
                                    lambda_=0.1,
                                    inverser=False)
    models_updated = models + [("new_%d" % (i,), es)
                               for i, m in enumerate(models_new)]

    Y = build_Y(X, models_updated, n_classes)
    es = EnsembleMachine(n_components=2)
    es.fit_with_mds(Y)
    plot_model_embeddings(models_updated, es.Z.get_value(),
                          save_file="after.png")

    light.endings()
    light.store_experiment()
    light.close()
