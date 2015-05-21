import cPickle as pickle
from theano import config


def load(filename):
    obj = pickle.load(open(filename))
    return obj


def get_pred(obj, num_prediction=None, num_folds=None):
    o = obj["pred"]
    y_dim = obj["y_dim"]
    if num_prediction is None:
        num_prediction = obj["num_prediction"]
    if num_folds is None:
        num_folds = obj["pred"].shape[1]

    o = o.reshape( (o.shape[0], o.shape[1], o.shape[2] / y_dim, y_dim) )
    o = o[:, :, 0:num_prediction, :]
    o = o.reshape( (o.shape[0], o.shape[1], o.shape[2] * o.shape[3]) )
    o = o[:, 0:num_folds, :]
    o = o.reshape( (o.shape[0], o.shape[1]*o.shape[2]) )
    o = o.astype(config.floatX)
    return o

import pandas as pd
import numpy as np

def get_scores(obj, filename):
    scores_csv = pd.read_csv(filename)
    filter_names = np.array([False] * len(scores_csv))
    for name in obj["names"]:
        filter_names |= (scores_csv["model"]==name.split("#")[1])

    scores_csv = scores_csv[filter_names]
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
            only_model_names[ind[model_name]] = scores_csv["model"][ind]
    return scores_csv, scores, contrib

def construct_table(Z, scores, contrib, names):

    columns = ["x", "y", "score", "contrib", "name", "team", "tag", "team_id"]

    data = pd.DataFrame(columns=columns)

    data["x"] = Z[:, 0]
    data["y"] = Z[:, 1]
    data["score"] = scores
    data["contrib"] = contrib
    data["name"] = names
    data["team"] = [name.split("#")[0] for name in names]
    data["tag"] = [name.split("#")[1] for name in names]

    t = list(set(data["team"].tolist()))
    ind = {a: i for i, a in enumerate(t)}
    data["team_id"] = [ind[tname] + 1 for tname in data["team"]]
    return data


from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import hashlib
target_column_name = 'target'
train_filename = "train.csv"
test_filename = "test.csv"


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
        print(type(train_is))
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

def split_data(num_folds=27*3, skf_test_size=0.5, random_state=57):
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


def get_all_data(num_folds, FeatureExtractor=FeatureExtractor):
    X_all, y_all, skf = split_data(num_folds)
    skf = list(skf)
    skf = skf[0:num_folds]
    X_all = FeatureExtractor().transform(X_all).astype(config.floatX)
    s = StandardScaler()
    X_all = s.fit_transform(X_all)
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(y_all)
    X_all = X_all.astype(config.floatX)
    return X_all, y_all, skf, s, label_encoder


def get_test_data(X_all, y_all, skf, num_prediction=None):
    skf = list(skf)

    if num_prediction is None:
        num_prediction = X_all[skf[0][1]].shape[0]

    num_folds = len(skf)
    X = np.zeros( (num_prediction, X_all.shape[1], num_folds) ).astype(config.floatX)
    y = np.empty((X.shape[0], num_folds), dtype=object)
    fold = 0

    idx_rest = np.ones((X_all.shape[0],))

    for train, test in skf:
        X[:, :, fold] = X_all[test][0:num_prediction]
        y[:, fold] = y_all[test][0:num_prediction]
        fold += 1
        idx_rest[test] = False

    idx_rest = idx_rest.astype(np.bool)
    X = X.transpose((0, 2, 1))
    X = X.reshape( (X.shape[0]*X.shape[1], X.shape[2]) )
    y = y.flatten()
    y = y.astype('int32')
    return X, y

def get_models(pred, y_dim):
    models  = pred
    models = models.astype(config.floatX)
    models = models.reshape((models.shape[0], models.shape[1] / y_dim, y_dim))
    models = models.transpose((1, 0, 2))
    return models

