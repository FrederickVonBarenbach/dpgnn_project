import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    constants = {
        "device": "cuda",
        "batch_size": 10000,
        "delta": 0.00000005,
        "clipping_threshold": 0.005,
        "noise_multiplier": 2,
        "lr": 1e-3, 
        "weight_decay": 1e-5,
        "dataset": "ogb_mag"
    }
    filepath = "./data/results.csv"
    epsilon_bound = 15

    df = pd.read_csv(filepath)
    for (key, value) in df.items():
        df = df.loc[df[key] == value]
    df = df.loc[df["epsilon"] > epsilon_bound]

    graph_type = "heatmap"
    axes = ("degree_bound", "r_hop")
    value = "test_acc"

    if graph_type == "heatmap":
        df = df.groupby([axes[0], axes[1], "step"], as_index=False).mean(numeric_only=True)
        df = df.loc[df.groupby([axes[0], axes[1]])["epsilon"].idxmin()]
        df = pd.pivot_table(df, values=value, index=[axes[0]], columns=[axes[1]], aggfunc=np.mean)
        sns.heatmap(df, annot=True, fmt=".2%", vmax=0.3, vmin=0.22)
        plt.title(value + " heatmap")
        plt.savefig("./figs/heatmap.png")