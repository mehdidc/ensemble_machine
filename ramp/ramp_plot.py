# collect data
from lightexperiments.light import Light
light = Light(dict(read_only=True))
light.launch()
data = light.experiments.filter(
    lambda obj:("lambda" in obj and
                "tags" in obj and
                "sanity_check_big_lambda" in obj["tags"])
)
#for i in range(len(data)):
#    light.experiments.delete(data[i]["__id"])
light.close()
data = sorted(data, key=lambda d:d["lambda"])
print(len(data))
data = data[0::2]


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
for i, d in enumerate(data):
#    print(d["lambda"])
#    print(d["loss_ensemble_machine"])
    loss_ensemble_machine = (np.array(d["loss_ensemble_machine"]))
    loss_accuracy = np.array(d["loss_accuracy"])
    accuracy = np.array(d["accuracy"])
    loss = np.array(d["loss"])

    #loss_ensemble_machine = ((loss_ensemble_machine - loss_ensemble_machine.min())/
    #                        (loss_ensemble_machine.max() - loss_ensemble_machine.min()))
    #loss_accuracy = ((loss_accuracy - loss_accuracy.min())/
    #                       (loss_accuracy.max() - loss_accuracy.min()))
    #accuracy = ((accuracy - accuracy.min())/
    #            (accuracy.max() - accuracy.min()))


    plt.subplot(len(data), 2, i*2 + 1)
    plt.plot(loss_ensemble_machine, label="MDS loss", c='blue')
    plt.legend()
    plt.subplot(len(data), 2, i*2 + 2)
    plt.plot(accuracy, label="accuracy", c='red')
    plt.legend()
    #plt.plot(loss, label="loss")
    #plt.plot(accuracy)
#    plt.plot(d["loss_accuracy"])
#    plt.plot(d["accuracy"])
    plt.title("lambda=%F" % (d["lambda"]))
    plt.legend()
#plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.5)
#plt.figlegend( lines, labels, loc = 'lower center', ncol=5, labelspacing=0. )
plt.show()
