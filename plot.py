import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from configs.plot_config import *
from math import ceil

comparison_colors = ['blue', 'red', 'green', 'orange', 'purple']
value_styles = ['solid', 'dashed', 'dotted']

def plot_graph_line(df):
    df = df.loc[df["epsilon"] <= epsilon_bound]
    n_rows = len(list(variables["rows"].values())[0])
    n_columns = len(list(variables["columns"].values())[0])
    n_comparisons = len(list(comparisons.values())[0])

    # get y limits
    max_y = 0
    for i in range(len(values)):
        max_y = max(max_y, df[values[i]].max())
    max_y = 1.1*max_y

    fig, axs = plt.subplots(n_rows, n_columns, figsize=(2 + 4*n_columns, 1 + 4*n_rows))
    plt.suptitle(axes[1] + " vs " + axes[0] + " plot")
    fig.tight_layout()
    for row in range(n_rows):
        for column in range(n_columns):
            plt.subplot(n_rows, n_columns, row*n_columns + column + 1)
            df_setting = df
            # filter for given row and column
            for (key, value) in variables["rows"].items():
                df_setting = df_setting.loc[df_setting[key] == value[row]]
            for (key, value) in variables["columns"].items():
                df_setting = df_setting.loc[df_setting[key] == value[column]]
            # filter comparisons
            for comparison in range(n_comparisons):
                df_comparison = df_setting
                label_prefix = ""
                for (key, value) in comparisons.items():
                    df_comparison = df_comparison.loc[df_comparison[key] == value[comparison]]
                    label_prefix += value[comparison] + " "

                # get average according to the axes
                if axes[0] != "step":
                    df_comparison = df_comparison.groupby([axes[0], "step"], as_index=False).mean(numeric_only=True)
                else:
                    df_comparison = df_comparison.groupby(["step"], as_index=False).mean(numeric_only=True)
                x = df_comparison[axes[0]]
                # make plot
                for value in range(len(values)):
                    y = df_comparison[values[value]]
                    plt.plot(x, y, label=label_prefix + values[value], color=comparison_colors[comparison], linestyle=value_styles[value])
                    plt.xlabel(axes[0])
                plt.legend()
                plt.ylim((0, max_y))
            # subplot titles
            if row == 0:
                title = ""
                for (key, value) in variables["columns"].items():
                    title += key + ": " + str(value[column]) + ", "
                title = title[:-2]
                plt.text(0.5, 1.05, title, fontsize=12, 
                         horizontalalignment='center', verticalalignment='center', transform=axs[row, column].transAxes)
            if column == 0:
                plt.ylabel(axes[1])
                title = ""
                for (key, value) in variables["rows"].items():
                    title += key + ": " + str(value[row]) + ", "
                title = title[:-2]
                plt.text(-0.25, 0.5, title, fontsize=12, rotation=90,
                         horizontalalignment='center', verticalalignment='center', transform=axs[row, column].transAxes)
    fig.tight_layout()
    plt.savefig("./figs/graph.png")

if __name__ == '__main__':
    # constants = {
    #     "device": "cuda",
    #     "delta": 0.00000005,
    #     "clipping_threshold": 0.005, # how should I get C?
    #     "weight_decay": 1e-5,
    #     "setup": "ours",
    #     "encoder_dimensions": "[128, 256, 256]",
    #     "decoder_dimensions": "[256, 256, 349]",
    #     "dropout": 0.1,
    #     "dataset": "ogb_mag"
    # }
    # variables = ["lr", "noise_multiplier", "batch_size"]
    # filepath = "./data/results_backup.csv"
    # epsilon_bound = 8

    # df = pd.read_csv(filepath)
    # for (key, value) in constants.items():
    #     df = df.loc[df[key] == value]

    # graph_type = "best_heatmap"
    # axes = ("degree_bound", "r_hop")
    # values = ["test_acc", "train_acc"]

    df = pd.read_csv(filepath)
    for (key, value) in constants.items():
        df = df.loc[df[key] == value]

    if graph_type == "heatmap":

        df = df.loc[df["epsilon"] <= epsilon_bound]
        df = df.groupby([axes[0], axes[1], "step"], as_index=False).mean(numeric_only=True)
        df = df.loc[df.groupby([axes[0], axes[1]])["epsilon"].idxmax()]
        fig, axs = plt.subplots(ceil(len(values)/2), 2 if len(values) > 1 else 1)
        for i in range(len(values)):
            ax = axs[int(i/2), i%2]
            df_i = pd.pivot_table(df, values=values[i], index=[axes[0]], columns=[axes[1]], aggfunc=np.mean)
            sns.heatmap(df_i, annot=True, fmt=".2%", vmax=0.3, vmin=0.22, ax=ax)
            ax.title(values[i] + " heatmap")
        plt.savefig(storepath + "heatmap.png")

    elif graph_type == "best_heatmap":
        
        df = df.loc[df["epsilon"] <= epsilon_bound]
        df = df.groupby([axes[0], axes[1], "step", *(variables["best"])], as_index=False).mean(numeric_only=True)
        df = df.loc[df.groupby([axes[0], axes[1]])["epsilon"].idxmax()]
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df)
        df_i = pd.pivot_table(df, values=values[0], index=[axes[0]], columns=[axes[1]], aggfunc=np.max)
        sns.heatmap(df_i, annot=True, fmt=".2%", vmax=0.3, vmin=0.22)
        plt.title(values[0] + " heatmap")
        plt.tight_layout()
        plt.savefig("./figs/best_heatmap.png")

    elif graph_type == "line":
        plot_graph_line(df)