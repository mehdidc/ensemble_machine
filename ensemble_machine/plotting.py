from __future__ import division
import matplotlib.pyplot as plt

import numpy as np

def plot_model_embeddings(models, Z, title="untitled", save_file=None):

    plt.ion()

    x = Z[:, 0]
    y = Z[:, 1]
    radii = 300.
    names = [name for name, model in models]

    models_kinds = [name.split("_")[0] for name in names]
    kinds = list(set(models_kinds))
    mapping = {k: i + 1 for i, k in enumerate(kinds)}
    colors = [mapping[k] for k in models_kinds]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=radii,
               c=colors, alpha=0.5, picker=True)

    annotations = []
    for name, x_i, y_i in zip(names, x, y):
        a = ax.annotate(name,
                     xy=(x_i, y_i),
                     xytext=(-30, 30),
                     textcoords='offset points',
                     ha='right',
                     va='bottom',
                     bbox=dict(boxstyle='round,pad=1.', fc='yellow', alpha=0.5),
                     arrowprops = dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        #a.set_visible(False)
        annotations.append(a)

    def onpick3(event):
        for a in annotations:
            a.set_visible(False)
        for ind in event.ind:
            annotations[ind].set_visible(True)
        fig.canvas.draw()
        return True

    fig.canvas.mpl_connect('pick_event', onpick3)
    if save_file is not None:
        plt.savefig(save_file)
    plt.show()
