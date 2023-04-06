import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from configs.plot_config import *
from math import ceil

comparison_colors = ['blue', 'red', 'green', 'orange', 'purple']
value_styles = ['solid', 'dashed', 'dotted']

def plot_heatmap(df):
    df = df.loc[df["epsilon"] <= epsilon_bound]
    n_rows = len(list(variables["rows"].values())[0]) if "rows" in variables else 1
    n_columns = len(list(variables["columns"].values())[0]) if "columns" in variables else 1

    # make subplots
    fig, axs = plt.subplots(n_rows, n_columns, figsize=(2 + 6*n_columns, 1 + 6*n_rows))
    if n_columns == 1 and n_rows == 1:
        axs = np.array([[axs]])
    else:
        axs = np.reshape(axs, (n_rows, n_columns))
    plt.suptitle(values[0] + " heatmap")
    fig.tight_layout()
    # plot each of the subplots
    for row in range(n_rows):
        for column in range(n_columns):
            plt.subplot(n_rows, n_columns, row*n_columns + column + 1)
            df_setting = df
            # filter for given row and column
            if "rows" in variables:
                for (key, value) in variables["rows"].items():
                    df_setting = df_setting.loc[df_setting[key] == value[row]]
            if "columns" in variables:
                for (key, value) in variables["columns"].items():
                    df_setting = df_setting.loc[df_setting[key] == value[column]]
            # make plot
            df_setting = df_setting.groupby([axes[0], axes[1], "step", *(variables["best"])], as_index=False).mean(numeric_only=True)
            df_setting = df_setting.loc[df_setting.groupby([axes[0], axes[1], *(variables["best"])])["epsilon"].idxmax()]
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print((df_setting.loc[df_setting.groupby([axes[0], axes[1]])[values[0]].idxmax()])[[axes[0], axes[1], *(variables["best"]), values[0]]])
            df_setting = pd.pivot_table(df_setting, values=values[0], index=[axes[0]], columns=[axes[1]], aggfunc=np.max)
            sns.heatmap(df_setting, annot=True, fmt=".2%", vmax=value_range[1], vmin=value_range[0], cmap="RdYlGn")
            # subplot titles
            if row == 0 and "columns" in variables:
                title = ""
                line = 0
                for (key, value) in variables["columns"].items():
                    entry = key + ": " + str(value[column]) + ", "
                    line += len(entry)
                    if (line > 26): # if line is longer than 26 chars, add new line
                        title += "\n"
                        line = 0
                    title += entry
                title = title[:-2]
                plt.text(0.5, 1.05, title, fontsize=12, 
                         horizontalalignment='center', verticalalignment='center', transform=axs[row, column].transAxes)
            if column == 0 and "rows" in variables:
                plt.ylabel(axes[1])
                title = ""
                for (key, value) in variables["rows"].items():
                    title += key + ": " + str(value[row]) + ", "
                title = title[:-2]
                plt.text(-0.25, 0.5, title, fontsize=12, rotation=90,
                         horizontalalignment='center', verticalalignment='center', transform=axs[row, column].transAxes)
    fig.tight_layout()
    plt.savefig(storepath)


def plot_graph_line(df):
    df = df.loc[df["epsilon"] <= epsilon_bound]
    n_rows = len(list(variables["rows"].values())[0]) if "rows" in variables else 1
    n_columns = len(list(variables["columns"].values())[0]) if "columns" in variables else 1
    n_comparisons = len(list(comparisons.values())[0])

    # make subplots
    fig, axs = plt.subplots(n_rows, n_columns, figsize=(2 + 4*n_columns, 1 + 4*n_rows))
    if n_columns == 1 and n_rows == 1:
        axs = np.array([[axs]])
    else:
        axs = np.reshape(axs, (n_rows, n_columns))
    plt.suptitle(axes[1] + " vs " + axes[0] + " plot")
    fig.tight_layout()
    # plot each of the subplots
    for row in range(n_rows):
        for column in range(n_columns):
            plt.subplot(n_rows, n_columns, row*n_columns + column + 1)
            df_setting = df
            # filter for given row and column
            if "rows" in variables:
                for (key, value) in variables["rows"].items():
                    df_setting = df_setting.loc[df_setting[key] == value[row]]
            if "columns" in variables:
                for (key, value) in variables["columns"].items():
                    df_setting = df_setting.loc[df_setting[key] == value[column]]
            # filter comparisons
            for comparison in range(n_comparisons):
                df_comparison = df_setting
                label_prefix = ""
                for (key, value) in comparisons.items():
                    df_comparison = df_comparison.loc[df_comparison[key] == value[comparison]]
                    label_prefix += str(value[comparison]) + " "

                # filter best value
                df_best = df_comparison.groupby(["step", *(variables["best"])], as_index=False).mean(numeric_only=True)
                df_best = df_best.loc[df_best.groupby([*(variables["best"])])["epsilon"].idxmax()]
                best_values = (df_best.loc[df_best[values[0]].idxmax()])[[*(variables["best"])]]
                for variable in variables["best"]:
                    df_comparison = df_comparison.loc[df_comparison[variable] == best_values[variable]]

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
                plt.ylim((value_range[0], value_range[1]))
            # subplot titles
            if row == 0 and "columns" in variables:
                title = ""
                line = 0
                for (key, value) in variables["columns"].items():
                    entry = key + ": " + str(value[column]) + ", "
                    line += len(entry)
                    if (line > 26): # if line is longer than 26 chars, add new line
                        title += "\n"
                        line = 0
                    title += entry
                title = title[:-2]
                plt.text(0.5, 1.05, title, fontsize=12, 
                         horizontalalignment='center', verticalalignment='center', transform=axs[row, column].transAxes)
            if column == 0 and "rows" in variables:
                plt.ylabel(axes[1])
                title = ""
                for (key, value) in variables["rows"].items():
                    title += key + ": " + str(value[row]) + ", "
                title = title[:-2]
                plt.text(-0.25, 0.5, title, fontsize=12, rotation=90,
                         horizontalalignment='center', verticalalignment='center', transform=axs[row, column].transAxes)
    fig.tight_layout()
    plt.savefig(storepath)

if __name__ == '__main__':
    if using_wandb == True:
        import pandas as pd 
        import wandb
        api = wandb.Api()

        # Project is specified by <entity/project-name>
        runs = api.runs(filepath)

        df = pd.DataFrame()
        i = 0
        n = len(runs)
        for run in runs: 
            print(str(i) + "/" + str(n), end="\r")

            if run.state != "finished":
                i += 1
                continue

            # save the metrics for the run to a csv file
            metrics_df = run.history()

            # .config contains the hyperparameters.
            #  We remove special values that start with _.

            config_list = [{k: v for k,v in run.config.items() if not k.startswith('_')}] * metrics_df.shape[0]
            config_df = pd.DataFrame(config_list).drop(columns="epsilon")

            run_df = pd.concat([metrics_df, config_df], axis=1)

            df = pd.concat([df, run_df])
            i += 1
        df.to_csv("data/"+filepath.replace("/", "_")+".zip", index=False, compression={"method": "zip", 
                                                                               "archive_name": filepath.replace("/", "_")+"_data.csv"}) 
    else:
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
        plt.savefig(storepath)

    elif graph_type == "best_heatmap":
        plot_heatmap(df)

    elif graph_type == "line":
        plot_graph_line(df)