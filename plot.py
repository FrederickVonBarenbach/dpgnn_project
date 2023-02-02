import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    df = pd.read_csv("results_exp1.csv")
    df = pd.pivot_table(df, values='accuracy', index=['degree_bound'], columns=['r_hop'], aggfunc=np.mean)
    sns.heatmap(df, annot=True, fmt=".2f")
    plt.title("accuracy heatmap")
    plt.savefig("heatmap.png")