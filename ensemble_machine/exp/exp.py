import os
from collections import OrderedDict
import numpy as np
from sklearn.cross_validation import train_test_split
from lasagne.generative.neural_net import NeuralNet
from lasagne.easy import (BatchOptimizer, LightweightModel)
from lasagne import layers, nonlinearities, updates, init

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

import theano.tensor as T
import pandas

from hp_toolkit.hp import Param, parallelizer, minimize_fn_with_hyperopt, find_best_hp
from utils import logloss
from wrappers.py_neural import PyneuralWrapper
from wrappers.neural_net import NeuralNetWrapper
from wrappers.bagging import Bagging 
from wrappers.adaboost import AdaBoost
from wrappers.repulsive_neural_net import RepulsiveNeuralNet

from lightexperiments.light import Light


def launch(X, y, seed=100):
    light = Light()
    np.random.seed(seed)
    light.set_seed(seed)

    X, y = shuffle(X, y)
    #X=X[0:100]
    #y=y[0:100]


    test_ratio = 0.25
    valid_ratio = 0.25
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=test_ratio)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=valid_ratio)

    # init & prepare
    Baseline, Best = "baseline", "best"
    neural_net_hp = dict()
    eval_functions = {"logloss": lambda model, X, y: float(logloss(model.predict_proba(X), y)),
            "accuracy": lambda model, X, y: float((model.predict(X)==y).mean()) }
    neural_net_hp[Baseline] = dict(
            num_units=100,
            nb_layers=1,
            batch_size=256,
            learning_rate=0.1
    )
    nb_models_per_ensemble = 10
    nb_models_per_ensemble_for_neural_net = 10
    max_evaluations_hp = 20 
    big_neural_net = True 
    bagging = True
    adaboost = True
    repulsive = True
    light.set("nb_models_per_ensemble", nb_models_per_ensemble)
    light.set("nb_models_per_ensemble_for_neural_net", nb_models_per_ensemble_for_neural_net)
    light.set("max_evaluations_hp", max_evaluations_hp)
    light.set("valid_ratio", valid_ratio)
    light.set("test_ratio", test_ratio)
    models_stats = OrderedDict()

    #1) find best neural net architecture
    best_hp, best_score = find_best_hp(NeuralNetWrapper,
                              (minimize_fn_with_hyperopt),
                              X_train,
                              X_valid,
                              y_train,
                              y_valid,
                              max_evaluations=max_evaluations_hp)
    neural_net_hp[Best] = best_hp
    neural_net_best_score = best_score
    light.set("neural_net_hp", neural_net_hp)
    light.set("neural_net_best_score", neural_net_best_score)
    # prepare big_neural_net
    if big_neural_net is True:
        big_neural_net_hp = dict()
        for hp_name, hp_value in neural_net_hp.items():
            big_neural_net_hp[hp_name] = hp_value.copy()
            big_neural_net_hp[hp_name]["num_units"] = hp_value["num_units"] * nb_models_per_ensemble_for_neural_net
        #2) try a Big neural net
        big_neural_net_models = dict()
        for name, hp in big_neural_net_hp.items():
            nnet = NeuralNetWrapper(**hp)
            nnet.fit(X_train_full, y_train_full,
                     X_valid=X_test, y_valid=y_test,
                     eval_functions=eval_functions)
            big_neural_net_models[name] = nnet
        models_stats["big_neural_net"] = {name:model.stats for name, model in big_neural_net_models.items()}
    
    #3) try bagging with the best and baseline architecture
    if bagging is True:
        bagging_models = dict()
        for name, hp in neural_net_hp.items():
            bagging = Bagging(base_estimator=NeuralNetWrapper(**hp), 
                              n_estimators=nb_models_per_ensemble)
            bagging.fit(X_train_full, y_train_full, 
                        X_valid=X_test, y_valid=y_test,
                        eval_functions=eval_functions)
            bagging_models[name] = bagging
        models_stats["bagging"] = {name:model.stats for name, model in bagging_models.items()}

    #4) try adaboost
    if adaboost is True:
        adaboost_models = dict()
        for name, hp in neural_net_hp.items():
            adaboost = AdaBoost(base_estimator=NeuralNetWrapper(**hp),
                                n_estimators=nb_models_per_ensemble)
            adaboost.fit(X_train_full, y_train_full, 
                         X_valid=X_test, y_valid=y_test,
                         eval_functions=eval_functions)
            adaboost_models[name] = adaboost
        models_stats["adaboost"] = {name:model.stats for name, model in adaboost_models.items()}
    
    #5)  find best lambda for repulsive neural net
    if repulsive is True:
        lambdas = dict()
        for name, hp in neural_net_hp.items():
            class RepulsiveNeuralNetBis(RepulsiveNeuralNet):
                def __init__(self, **params):
                    super(RepulsiveNeuralNetBis, self).__init__(**params)
                    self.__dict__.update(hp)
                    self.ensemble_size = nb_models_per_ensemble_for_neural_net 
            # optimize only lambda_
            best_hp, best_score = find_best_hp(RepulsiveNeuralNetBis,
                                      minimize_fn_with_hyperopt,
                                      X_train,
                                      X_valid,
                                      y_train,
                                      y_valid,
                                      allowed_params=["lambda_"],
                                      max_evaluations=max_evaluations_hp)
            rep = RepulsiveNeuralNetBis()
            rep.lambda_ = 0.
            rep.fit(X_train, y_train)
            score = (rep.predict(X_valid)!=y_valid).mean()
            if score < best_score:
                lambdas[name] = 0.
            else:
                lambdas[name] = best_hp.get("lambda_")
        light.set("lambdas", lambdas)
        #6) then retrain repulsive nets with best lambdas
        repulsive_neural_net_models = dict()
        for name, hp in neural_net_hp.items():
            hp_ = hp.copy()
            hp_["lambda_"] = lambdas[name]
            repulsive_neural_net = RepulsiveNeuralNet(**hp_)
            repulsive_neural_net.fit(X_train_full, y_train_full, 
                                     X_valid=X_test, y_valid=y_test,
                                     eval_functions=eval_functions)
            repulsive_neural_net_models[name] = repulsive_neural_net
        models_stats["repulsive_neural_net"] = {name:model.stats for name, model in repulsive_neural_net_models.items()}
    light.set("models_stats", models_stats)

def report_learning_curves(experiment, report_dir="report"):
    html = []

    html.append("<html>")
    html.append("<body>")
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    stats = experiment["models_stats"]

    fig = plt.figure()

    html.append("<p>Dataset : {0} </p>".format(experiment["dataset"]))
    cur_figure_id = 0
    figures = dict()

    for model_name, config in stats.items():
        html.append("<h1>{0}</h1>".format(model_name))
        for config_name, stats in config.items():
            html.append("<h2>Architecture : {0}</h2>".format(config_name))
            stat_names = stats[0].keys()
            stat_names = set([name.split("_")[0] for name in stat_names if "_" in name])
            for name in stat_names:
                #only logloss
                #if name not in ("logloss",):
                #    continue
                html.append("<h3>Criterion : {0}</h3>".format(name))
                values_train = [st[name+"_train"] for st in stats]
                values_valid = [st[name+"_valid"] for st in stats]
                iterations = range(len(values_train))
                plt.clf()
                plt.plot(iterations, values_train, label="train", c='b')
                plt.plot(iterations, values_valid, label="valid", c='g')
                plt.title("{0} {1} curves".format(model_name, name))
                plt.xlabel("epochs")
                plt.ylabel(name)
                plt.legend()
                filename = "{0}_{1}_{2}.png".format(model_name, config_name, name)
                plt.savefig(os.path.join(report_dir, filename))
                html.append("<img src='{0}'></img>".format(filename))
                cur_figure_id += 1
                html.append("<p>Figure {0}</p>".format(cur_figure_id))
                figures[(model_name, config_name, name)] = cur_figure_id
                plt.show()
        html.append("<hr />")

    stats = experiment["models_stats"]
    architectures = stats.values()[0].keys()

    for name in ("logloss_train","accuracy_train", "logloss_valid", "accuracy_valid"):
        for a in architectures:
            plt.clf()
            for model_name, config in stats.items():
                    st = config[a]
                    try:
                        L = [s[name] for s in st]
                    except:
                        continue
                    iterations = range(len(L))
                    plt.plot(iterations, L, label=model_name)
            plt.xlabel("epochs")
            plt.ylabel(name)
            plt.legend()
            filename = "all-{0}-{1}.png".format(a, name)
            plt.savefig(os.path.join(report_dir, filename))
            html.append("<h1>All {0} scores with {1}</h1>".format(name, a))
            html.append("<img src='{0}'></img>'".format(filename))


    html.append("<p><strong>Duration</strong> : {0:.2f} sec</p>".format(experiment.get("duration")))
    html.append("<p><strong>Started at </strong>{0}</p>".format(experiment.get("start")))
    html.append("<p><strong>Ended at </strong>{0}</p>".format(experiment.get("end")))
    html.append("<p><strong>Seed</strong> : {0}</p>".format(experiment.get("seed")))
    e = experiment
    story = [
            """the dataset was first divided into training and test set, ratio of test set is {0}""".format(e["test_ratio"]),
            """For hyper-parameter optimization, the full training set was divided into training and validation, the ratio of valid set was {0}""".format(e["valid_ratio"]),
            """baseline neural net hyper-parameters are : {0}""".format(e["neural_net_hp"]["baseline"]),
            """First find best neural net architecture, the number of evalutions for
                hyper-parameter optimization was : {0}, the best hyper-parameters found were : {1}""".format(e["max_evaluations_hp"],
                                                                                                             e["neural_net_hp"]["best"]),
            """then train a big neural net which is the same as baseline and 
               best but where the number of units are multiplied by 
               the size of the ensemble which is {0}""".format(e["nb_models_per_ensemble_for_neural_net"]),
            """then train bags of neural nets with baseline and best architecture, 
               the number of iterations was {0}""".format(e["nb_models_per_ensemble"]),
            """then train adaboost of neural nets with baseline and best architecture, 
               the number of iterations was {0}""".format(e["nb_models_per_ensemble"]),
            """then try repulsive neural nets, first take baseline and best architecture, then find best lambda for both of them.
               lambda for baseline was {0}, lambda for best was {1}.""".format(e["lambdas"]["baseline"], e["lambdas"]["best"]),
            """then retrain repulsive neural nets with the best lambdas found with baseline and best architecture, the other hyper-paremeters
               being exactly the same than baseline and best, the size of the ensemble was {0}""".format(e["nb_models_per_ensemble_for_neural_net"])
    ]
    html.append("<h1>Story</h1>")
    story_s = ["<p>{0}</p>".format(s) for s in story]
    html.extend(story_s)
    html.append("</body>")
    html.append("</html>")
    fd = open(os.path.join(report_dir, "index.html"), "w")
    fd.write("\n".join(html))
    fd.close()

if __name__ == "__main__":
    from datasets import datasets
    import os
    light = Light()
    light.launch()
     
    #r = list(light.db.find({"tags": "auto_ensemble_experiment"}))
    #for i in range(len(r)):
    #    try:
    #        os.mkdir("reports/report_{0}".format(i + 1))
    #    except Exception:
    #        pass
    #    report_learning_curves(r[i], "reports/report_{0}".format(i + 1))
    #light.close()
    #sys.exit(0)

    light.initials()
    light.tag("auto_ensemble_experiment")

    ds = "otto"
    light.set("dataset", ds)
    X, y = datasets.get(ds)()
    launch(X, y)
    light.endings()
    report_learning_curves(light.cur_experiment, "report_{0}".format(ds))
    light.store_experiment()
    light.close()
