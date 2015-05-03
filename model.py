import mds
import theano
import theano.tensor as T
from lasagne import easy, init, updates
import numpy as np

def ensemble_machine_model_loss(y_new, z_new, y_old, z_old, dist_y_func, dist_z_func):
    # y_new : (nb new models, nb outputs) predicted points by the new models
    # z_new : (nb new models, embedding dim) wanted points in the
    #         embedding
    # y_old : (nb models, nb outputs)
    # z_old : (nb models, embedding dim)


    y = T.concatenate([y_new, y_old], axis=0)
    dist_y = dist_y_func(y)

    z = T.concatenate([z_new, z_old], axis=0)
    dist_z = dist_z_func(z)

    return mds.MDS_loss(dist_y, dist_z)


class EnsembleMachine(object):

    def __init__(self, n_components=2,
                       dist_y=mds.euclidian_dist, dist_z=mds.euclidian_dist,
                       batch_optimizer=None):

        self.n_components = n_components
        if batch_optimizer is None:
            batch_optimizer = easy.BatchOptimizer()
        self.batch_optimizer = batch_optimizer
        self.dist_y = dist_y
        self.dist_z = dist_z

        self.Z = None

    def fit(self, Y):
        nb_examples = Y.shape[0]

        Z_init = np.random.uniform(size=(Y.shape[0], self.n_components))
        Z_init = Z_init.astype(theano.config.floatX)
        self.Z = theano.shared(Z_init, borrow=True)


        Y = theano.shared(Y, borrow=True)
        Y_batch = T.matrix('Y_batch')
        batch_index = T.iscalar('batch_index')
        batch_slice = easy.get_batch_slice(batch_index,
                                           self.batch_optimizer.batch_size)

        d_y = self.dist_y(Y_batch)
        d_z = self.dist_z(self.Z[batch_slice])

        loss = mds.MDS_loss(d_y, d_z)

        all_params = [self.Z]

        opti_function, opti_kwargs = self.batch_optimizer.optimization_procedure
        updates = opti_function(loss, all_params, **opti_kwargs)

        givens = {
            Y_batch: Y[batch_slice],
        }
        iter_train = theano.function(
            [batch_index], loss,
            updates=updates,
            givens=givens
        )
        nb_batches = easy.get_nb_batches(nb_examples, self.batch_optimizer.batch_size)
        self.batch_optimizer.optimize(nb_batches, iter_train)
        self.stress_ = theano.function([], loss, givens={Y_batch : Y, batch_index: 0})()
        self.Y = theano.shared(theano.function([], d_y, givens={Y_batch: Y})(), borrow=True)
        return self

    def update(self, X, models_new, Z_new, batch_optimizer=None):

        if batch_optimizer is None:
            batch_optimizer = easy.BatchOptimizer()

        X_batch = T.matrix()
        batch_index = T.iscalar('batch_index')
        batch_slice = easy.get_batch_slice(batch_index,
                                           batch_optimizer.batch_size)

        # X : dataset (nb examples, nb features)
        # Z_new (nb_new_models, nb_components)
        # models_new : list of nb_new_models LightweightModel
        outputs = []
        for model in models_new:
            output = model.get_output(X_batch).flatten().dimshuffle('x', 0)
            outputs.append(output) # (1, nb examples * nb_outputs)

        Y_new =  T.concatenate(outputs, axis=0) # (nb_new_models, nb_examples * nb_outputs)

        loss = ensemble_machine_model_loss(Y_new, Z_new, self.Y, self.Z, self.dist_y, self.dist_z)
        all_params = list(set(model.get_all_params() for model in models_new))

        opti_function, opti_kwargs = batch_optimizer.optimization_procedure
        updates = opti_function(loss, all_params, **opti_kwargs)

        X = theano.shared(X, borrow=True)
        givens = {
            X_batch: X[batch_slice],
        }
        iter_train = theano.function(
            [batch_index], loss,
            updates=updates,
            givens=givens
        )
        nb_batches = easy.get_nb_batches(nb_examples, self.batch_optimizer.batch_size)
        self.batch_optimizer.optimize(nb_batches, iter_train)
        return self

def build_Y(X, models):
    Y = np.zeros((len(models), X.shape[0]))
    for i, (name, model) in enumerate(models):
        Y[i, :] = model.predict(X)
    return Y

if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.datasets import make_classification

    import matplotlib.pyplot as plt
    from plotting import plot_model_embeddings

    X, y = make_classification(100)
    X = X.astype(theano.config.floatX)


    models = [
        ('rf', RandomForestClassifier()),
        ('ada', AdaBoostClassifier()),
        ('logreg', LogisticRegression()),
        ('svm', SVC()),
    ]

    for name, model in models:
        model.fit(X, y)
    Y = build_Y(X, models)
    Y = Y.astype(theano.config.floatX)

    batch_optimizer = easy.BatchOptimizer(verbose=1,
                                          batch_size=Y.shape[0],
                                          max_nb_epochs=100,
                                          optimization_procedure=(updates.Adam, {"learning_rate": 0.3}))
    es = EnsembleMachine(batch_optimizer=batch_optimizer)
    es.fit(Y)

    plot_model_embeddings(models, es.Z.get_value())
    """
    x_in = layers.InputLayer(shape=(None, X.shape[1]))
    h = layers.DenseLayer(x_in, num_units=z_dim,
                          W=init.GlorotUniform(),
                          nonlinearity=nonlinearities.rectify)
    z_out = layers.DenseLayer(h, num_units=z_dim,
                              W=init.GlorotUniform(),
                              nonlinearity=nonlinearities.rectify)
    nnet_x_to_z = LightweightModel([x_in],
                                   [z_out])

    models = [model]

    es.update()
    """
