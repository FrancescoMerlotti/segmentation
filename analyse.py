import os
import argparse
import pandas as pd
from scipy.stats import skew
import matplotlib.pyplot as plt

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
# load dataframe
df = pd.read_csv(filename)
# compute results
results = compute_stats(df, 'area')
for key, value in results.items(): print(f"{key:10}= {value:8.2f}")

fig, ax = plt.subplots(1,1,figsize=(6,6))
ax.hist(df['area'], bins=50)
ax.set_xlabel('Size')
ax.set_title('Size distribution')
if not os.path.isdir('plots'): os.mkdir('plots')
fig.savefig('plots/'+filename.split('/')[1].split('.')[0]+'.pdf')

