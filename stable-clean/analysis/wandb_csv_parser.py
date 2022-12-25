from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import click


# @click.argument('path', default=None)
def main():
    plot_multiple_runs(os.path.join("output/avg_reward.csv"))

def plot_multiple_runs(path, benchmark=None, name=None):
    database = pd.read_csv(path)
    print(database)
    x_axis = "epoch"
    y_axis = ["constant", "noise", "base"]

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig = plt.figure(figsize=((3/4)*7, (3/4)*4.7))
    plt.grid()
    fontsize=16
    for y in y_axis:
        plt.plot(database[x_axis], database[y], label=y)
    plt.legend(fontsize=18, loc='lower right')
    plt.xlabel("Epoch $n$", fontsize=fontsize)
    plt.ylabel("Average Return $R$", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(name, fontsize=18)
    #fig.axes[0].set_xticks(np.arange(20000, 250000, 50000)) # for cheetah
    #plt.xscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join("figures/avg_reward.pdf"), dpi=300, format="pdf")

if __name__ == "__main__":
    main()