import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import skew
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'sans-serif',})

def compute_stats(df, column_name):
    data = df[column_name]
    # mean, median, mode
    stats = {"mean": data.mean(), "median": data.median(), "mode": data.mode()[0],}
    # interquartile
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    stats["iqr"] = q3 - q1
    # std deviation
    stats["std"] = data.std()
    # # skewness
    # stats["skewness"] = skew(data)
    # 95th percentile
    stats["95th"] = data.quantile(0.95)
    return stats

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, help='File to be analysed.')
args = parser.parse_args()

if not args.filename: raise ValueError('Missing argument! Provide the name of the file to analyse, e.g., `data/data-file.csv`.')
filename = args.filename
isRatio = '-µm' in filename
# load dataframe
df = pd.read_csv(filename)
# compute results
for diam in ['fer-diam', 'diag-diam']:
    results = compute_stats(df, diam)
    print(diam+' statistical summary:')
    for key, value in results.items(): print(f"\t{key:10}= {value:8.2f} µm")

grid = {'diam': (0,0), 'max-diam': (0,1), 'fer-diam': (1,0), 'diag-diam': (1,1),}

fig, axs = plt.subplots(2,2,figsize=(12,6),sharex=True,sharey=True)
add = ' [µm]' if isRatio else ''
radii = np.array([df[rad] for rad in grid]).flatten()
bins = np.linspace(0., np.max(radii), 50)
for rad in grid:
    axs[grid[rad]].hist(df[rad], bins=bins,)
    axs[grid[rad]].set_xlabel(rad+add, fontdict={'size': 12})

fig.suptitle(filename.split('/')[1].split('.')[0], fontsize=20)
for ax in axs.flatten():
    ax.tick_params(axis="both", which="both", direction="in", labelsize=10)
    ax.grid(alpha=0.2, linestyle='--')
    # ax.legend(frameon=False, markerfirst=False)
if not os.path.isdir('plots'): os.mkdir('plots')
fig.savefig('plots/'+filename.split('/')[1].split('.')[0]+'.pdf')

