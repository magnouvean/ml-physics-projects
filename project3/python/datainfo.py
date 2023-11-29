import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../data/creditcard_2023.csv").sample(frac=1)

X = df.drop(["id", "Class"], axis=1)
y = df[["Class"]]

print(f"Total observations: {X.shape[0]}")
print(f"Number of non-fraud: {np.sum(y==0, axis=0)}")
print(f"Number of fraud: {np.sum(y==1, axis=0)}")

fig, ax = plt.subplots(6, 5)
for i, col in enumerate(X.columns):
    r = i // 5
    c = i % 5
    sns.boxplot(
        y=col,
        x="Class",
        data=df[:10000],
        ax=ax[r, c],
        showfliers=False,
    )
    ax[r, c].set(yticklabels=[])
    ax[r, c].set_xlabel("")

ax[5, 4].set(yticklabels=[], xticklabels=[])
fig.subplots_adjust(hspace=0.8, wspace=0.5, bottom=0.15)
fig.savefig("../figures/data_features_boxplot.png", dpi=196)
