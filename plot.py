import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    constants = {
        "device": "cuda",
        "delta": 0.00000005,
        "clipping_threshold": 0.005, # how should I get C?
        "weight_decay": 1e-5,
        "setup": "ours",
        "encoder_dimensions": "[128, 256, 256]",
        "decoder_dimensions": "[256, 256, 349]",
        "dropout": 0.1,
        "dataset": "ogb_mag"
    }
    variables = ["lr", "noise_multiplier", "batch_size"]
    filepath = "./data/results.csv"
    epsilon_bound = 10

    df = pd.read_csv(filepath)
    for (key, value) in constants.items():
        df = df.loc[df[key] == value]
    df = df.loc[df["epsilon"] > epsilon_bound]

    graph_type = "best_heatmap"
    axes = ("degree_bound", "r_hop")
    value = "test_acc"



    if graph_type == "heatmap":

        df = df.groupby([axes[0], axes[1], "step"], as_index=False).mean(numeric_only=True)
        df = df.loc[df.groupby([axes[0], axes[1]])["epsilon"].idxmin()]
        df = pd.pivot_table(df, values=value, index=[axes[0]], columns=[axes[1]], aggfunc=np.mean)
        sns.heatmap(df, annot=True, fmt=".2%", vmax=0.3, vmin=0.22)
        plt.title(value + " heatmap")
        plt.savefig("./figs/heatmap.png")

    elif graph_type == "best_heatmap":
        
        df = df.groupby([axes[0], axes[1], "step", *variables], as_index=False).mean(numeric_only=True)
        df = df.loc[df.groupby([axes[0], axes[1]])["epsilon"].idxmin()]
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df)
        df = pd.pivot_table(df, values=value, index=[axes[0]], columns=[axes[1]], aggfunc=np.max)
        sns.heatmap(df, annot=True, fmt=".2%", vmax=0.3, vmin=0.22)
        plt.title(value + " heatmap")
        plt.savefig("./figs/best_heatmap.png")

    