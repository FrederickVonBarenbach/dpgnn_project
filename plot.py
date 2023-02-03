import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    df = pd.read_csv("results_original_exp1.csv")
    df = pd.pivot_table(df, values='accuracy', index=['degree_bound'], columns=['r_hop'], aggfunc=np.mean)
    sns.heatmap(df, annot=True, fmt=".2f", vmax=30, vmin=22)
    plt.title("accuracy heatmap of original")
    plt.savefig("org_heatmap.png")