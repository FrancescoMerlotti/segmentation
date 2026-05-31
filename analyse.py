import os
import argparse
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
    # skewness
    stats["skewness"] = skew(data)
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
results = compute_stats(df, 'area')
for key, value in results.items(): print(f"{key:10}= {value:8.2f}")

fig, axs = plt.subplots(1,2,figsize=(12,6),sharey=True)
axs[0].hist(df['area'], bins=25, label=filename.split('/')[1].split('.')[0])
axs[1].hist(df['radius'], bins=25, label=filename.split('/')[1].split('.')[0])
if isRatio:
    axs[0].set_xlabel('Area [µm$^2$]', fontdict={'size': 12})
    axs[1].set_xlabel('Radius [µm]', fontdict={'size': 12})
else:
    axs[0].set_xlabel('Area', fontdict={'size': 12})
    axs[1].set_xlabel('Radius', fontdict={'size': 12})
axs[0].set_title('Area distribution', fontdict={'size': 20})
axs[1].set_title('Radius distribution', fontdict={'size': 20})
for ax in axs:
    ax.tick_params(axis="both", which="both", direction="in", labelsize=10)
    ax.grid(alpha=0.2, linestyle='--')
    ax.legend(frameon=False, markerfirst=False)
if not os.path.isdir('plots'): os.mkdir('plots')
fig.savefig('plots/'+filename.split('/')[1].split('.')[0]+'.pdf')

