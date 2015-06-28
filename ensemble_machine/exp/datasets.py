import numpy as np
from sklearn.datasets import make_classification, fetch_covtype
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import shuffle
import pandas as pd


def build_make_classification():
    X, y = make_classification(n_samples=1000, n_features=20,
                               n_informative=4, n_redundant=2, n_classes=3)
    X = X.astype(np.float32)
    y = y.astype(np.int32)
    return X, y


def build_otto():
    X = pd.read_csv("otto/train.csv")
    y = X["target"]
    X = X.drop(["id", "target"], axis=1)
    X = np.array(X.values)
    y = np.array(y.values)
    y = LabelEncoder().fit_transform(y)
    y = y.astype(np.int32)
    X = np.log(2 + X)
    X = StandardScaler().fit_transform(X)
    X = X.astype(np.float32)
    return X, y


def build_covertype():
    data = fetch_covtype()
    X, y = data["data"], data["target"]
    X = StandardScaler().fit_transform(X)
    X = X.astype(np.float32)
    y = LabelEncoder().fit_transform(y)
    y = y.astype(np.int32)
    return X, y


def build_segment():
    data = pd.read_csv("segment/segment.dat", sep=" ")
    data = np.array(data.values)
    X, y = data[:, 0:-1], data[:, -1]
    X = StandardScaler().fit_transform(X)
    X = X.astype(np.float32)
    y = LabelEncoder().fit_transform(y)
    y = y.astype(np.int32)
    return X, y


datasets = dict(
    make_classification=build_make_classification,
    otto=build_otto,
    covertype=build_covertype,
    segment=build_segment
)
