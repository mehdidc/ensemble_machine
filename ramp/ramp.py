
# coding: utf-8

# In[9]:

import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import MDS, Isomap, TSNE, LocallyLinearEmbedding
import pandas as pd
from scipy.spatial.distance import euclidean, hamming
from ensemble_machine.model import EnsembleMachine
import theano

from lightexperiments import light
plt.ion()
light = light.Light()

seed = 128
np.random.seed(seed)
light.set_seed(seed)


light.launch()

light.initials()
light.file_snapshot()


ramp = 4


light.tag("ramp_%d_ensemble_machine" % (ramp,))

# In[2]:

obj = pickle.load(open("result_ramp%d.dat" % (ramp,)))


# In[3]:

obj.keys()


# In[4]:

num_folds = 1
num_prediction = obj["num_prediction"]
y_dim = obj["y_dim"]
o = obj["pred"]
o = o[:, 0:num_folds, :]
o = o.reshape( (o.shape[0], o.shape[1]*o.shape[2]) )
o = o.astype(theano.config.floatX)

es = EnsembleMachine(n_components=2)
es.fit_with_mds(o)
light.set("stress", es.stress_)
print(es.stress_)
Z = es.Z.get_value()
Y = es.Y.get_value()

# In[5]:

scores_csv = pd.read_csv("scores_ramp%d.csv" % (ramp,))
scores = [0] * len(obj["names"])
contrib = [0] * len(obj["names"])
only_model_names = [""] * len(obj["names"])
ind = {}
for i, name in enumerate(obj["names"]):
    ind[name] = i
for i, (model_name, s, c) in enumerate(zip(scores_csv["team"]+"#"+scores_csv["model"],
                            scores_csv["score"],
                            scores_csv["contributivity"])):
    if model_name in ind:
        scores[ind[model_name]] = s
        contrib[ind[model_name]] = c
        only_model_names[ind[model_name]] = scores_csv["model"][i]
scores = np.array(scores)
contrib = np.array(contrib)


# In[10]:

columns = ["x", "y", "score", "contrib", "name", "team", "tag", "team_id"]

data = pd.DataFrame(columns=columns)

data["x"] = Z[:, 0]
data["y"] = Z[:, 1]
data["score"] = scores
data["contrib"] = contrib
data["name"] = obj["names"]
data["team"] = [name.split("#")[0] for name in obj["names"]]
data["tag"] = [name.split("#")[1] for name in obj["names"]]

t = list(set(data["team"].tolist()))
ind = {a:i for i, a in enumerate(t)}
data["team_id"] = [ind[tname] + 1 for tname in data["team"]]

print(data)

# In[7]:

D = data[data["contrib"] > 2]
plt.scatter(D["x"], D["y"], s=D["contrib"]*10, c=D["team_id"], alpha=0.5)
for name, x, y in zip(D["tag"], D["x"], D["y"]):
    plt.annotate(name, xy=(x,y))
#plt.show()


# In[11]:

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import hashlib
target_column_name = 'target'
train_filename = "train.csv"
test_filename = "test.csv"
skf_test_size = 0.5
random_state = 57
n_CV = 27 * 3
def get_hash_string_from_indices(index_list):
    """We identify files output on cross validation (models, predictions)
    by hashing the point indices coming from an skf object.

    Parameters
    ----------
    test_is : np.array, shape (size_of_set,)

    Returns
    -------
    hash_string
    """
    hasher = hashlib.md5()
    hasher.update(index_list)
    return hasher.hexdigest()

def get_hashes(skf):
    h = []
    for train_is, test_is in skf:
        hash_string = get_hash_string_from_indices(train_is)
        h.append(hash_string)
    return h
# X is a list of dicts, each dict is indexed by column
def read_data(df_filename):
    df = pd.read_csv(df_filename, index_col=0) # this drops the id actually
    y_array = df[target_column_name].values
    X_dict = df.drop(target_column_name, axis=1).to_dict(orient='records')
    return X_dict, y_array

def prepare_data():
    pass
    # train and tes splits are given

def split_data(num_folds=n_CV):
    X_train_dict, y_train_array = read_data(train_filename)
    #X_test_dict, y_test_array = read_data(test_filename)
    skf = StratifiedShuffleSplit(y_train_array, n_iter=num_folds,
        test_size=skf_test_size, random_state=random_state)
    return X_train_dict, y_train_array, skf

class FeatureExtractor(object):

    def __init__(self):
        pass

    def fit(self, X_dict, y):
        pass

    def transform(self, X_dict):
        cols = X_dict[0].keys()
        return np.array([[instance[col] for col in cols] for instance in X_dict])

X_all, y_all, skf = split_data(num_folds)
skf = list(skf)
X_all = FeatureExtractor().transform(X_all).astype(theano.config.floatX)
s = StandardScaler()
X_all = s.fit_transform(X_all)

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_all = label_encoder.fit_transform(y_all)


# In[12]:

X = np.zeros( (  num_prediction, X_all.shape[1], num_folds) )
y = np.empty((X.shape[0], num_folds), dtype=object)
fold = 0

idx_rest = np.ones((X_all.shape[0],))

for train, test in skf:
    X[:, :, fold] = X_all[test][0:num_prediction]
    y[:, fold] = y_all[test][0:num_prediction]
    fold += 1
    idx_rest[test] = False
idx_rest = idx_rest.astype(np.bool)
X = X.reshape( (X.shape[0], X.shape[1]*X.shape[2]) )
y = y.flatten()


# In[13]:

X = X.astype(theano.config.floatX)
y = y.astype(np.int32)


# In[14]:

from lasagne import layers, init, nonlinearities, updates
from lasagne.easy import LightweightModel
import theano
from lasagne import easy
x_in = layers.InputLayer(shape=(None, X.shape[1]))
h = layers.DenseLayer(x_in, num_units=500,
                      W=init.GlorotNormal(),
                      nonlinearity=nonlinearities.rectify)
h = layers.DenseLayer(x_in, num_units=500,
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

light.tag("for_submission")
light.tag("sanity_check_big_lambda")

Z_new = [
    [data["x"][0], data["y"][0]] # brutal dropout
]
Z_new = np.array(Z_new, dtype=theano.config.floatX)

class MyBatchOptimizer(easy.BatchOptimizer):
    def iter_update(self, epoch, nb_batches,
                    iter_update_batch):
            super(MyBatchOptimizer, self).iter_update(epoch,
                                                      nb_batches,
                                                      iter_update_batch)
            #if epoch == self.max_nb_epochs - 1:
            #    print(es.get_dist_y(X))
            #    print(es.get_dist_z())

            loss_ensemble_machine = float(es.get_loss_ensemble_machine(X))
            loss_accuracy = float(es.get_loss_accuracy(X, y))
            accuracy = ((es.predict(X)==y).mean())
            loss = float(es.get_loss(X, y))
            light.append("loss_ensemble_machine", loss_ensemble_machine)
            light.append("loss_accuracy", loss_accuracy)
            light.append("loss", loss)
            light.append("accuracy", accuracy)

            print("loss ensemble machine : ", loss_ensemble_machine)
            print("loss accuracy : ", loss_accuracy)
            print("accuracy :", accuracy)
            print("loss :", loss)

procedure = (updates.momentum,
             {"learning_rate": 0.01, "momentum": 0.8})
batch_optimizer = MyBatchOptimizer(verbose=1,
                                   max_nb_epochs=1000,
                                   batch_size=X.shape[0],
                                   optimization_procedure=procedure)

# In[16]:
lambda_ = 0.1
light.set("lambda", lambda_)
es.update_with_gradient_descent(X, y, models_new, Z_new,
                                batch_optimizer=batch_optimizer,
                                lambda_=lambda_,
                                inverser=False)

plt.clf()
plt.plot(np.sqrt(np.array(light.cur_experiment["loss_ensemble_machine"])), label="EM")
plt.plot(light.cur_experiment["loss_accuracy"], label="ACC")
plt.legend()
#plt.show()

# In[62]:

new_models_predictions = [es.predict_proba(X).flatten()[np.newaxis, :] for model in models_new]
a=np.concatenate( [o] + new_models_predictions, axis=0)

es_2 = EnsembleMachine(n_components=2)
es_2.fit_with_mds(a)
Z = es_2.Z.get_value()


# In[63]:

data = pd.DataFrame(columns=columns)
scores_new = np.array(scores.tolist() + [0] * len(models_new))
contrib_new = np.array(contrib.tolist() + [10] * len(models_new))
names_new = obj["names"] + ["new_team#new"] * len(models_new)
data["x"] = Z[:, 0]
data["y"] = Z[:, 1]
data["score"] = scores_new
data["contrib"] = contrib_new
data["name"] = names_new
data["team"] = [name.split("#")[0] for name in names_new]
data["tag"] = [name.split("#")[1] for name in names_new]
t = list(set(data["team"].tolist()))
ind = {a:i for i, a in enumerate(t)}
data["team_id"] = [ind[tname] + 1 for tname in data["team"]]


# In[64]:

D = data[data["contrib"] > 2]
#D = data[data["team"].str.contains("NeuralTheano")]

plt.scatter(D["x"], D["y"], s=D["contrib"]*10, c=D["team_id"], alpha=0.5)
for name, x_, y_ in zip(D["tag"], D["x"], D["y"]):
    plt.annotate(name, xy=(x_,y_))

#plt.show()


# In[35]:

for i in range(len(models_new)):
    pred = getattr(es, "predict_%d" % (i,))
    print( (pred(X)==y).mean() )
    print((pred(X_all[idx_rest])))
    print((pred(X_all[idx_rest])==y_all[idx_rest]).mean())
    #(pred(X_all[idx_rest])==y[idx_rest]).mean()

light.endings()
light.store_experiment()
light.close()
