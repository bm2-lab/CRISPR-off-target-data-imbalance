import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# import seaborn as sns
# sns.set_style("whitegrid")

df = pd.read_csv('result/all_res.csv', index_col=None)

df_auc_fig = pd.DataFrame({'x': df['x'], 'y': df['auc'], 'h': df['h']})
df_pr_fig = pd.DataFrame({'x': df['x'], 'y': df['pr'], 'h': df['h']})
df_r_fig = pd.DataFrame({'x': df['x'], 'y': df['spearman'], 'h': df['h']})

xlst = ['Leave-sgRNA-Out', 'Leave-Study-Out: Common', 'Leave-Study-Out: Unique']
hlst = ['RF with balanced sampling', 'CRISTA', 'RF without balanced sampling','CFD','OptCD','CCTop']
corlst = ['blue', 'lime', 'red', 'magenta','gold', 'green']

dt_auc = {row[1]['h']:str(round(row[1]['y'],3)) for row in df_auc_fig[df_auc_fig['x']=='Leave-sgRNA-Out'].iterrows()}
dt_pr = {row[1]['h']:str(round(row[1]['y'],3)) for row in df_pr_fig[df_pr_fig['x']=='Leave-sgRNA-Out'].iterrows()}
dt_r = {row[1]['h']:str(round(row[1]['y'],3)) for row in df_r_fig[df_r_fig['x']=='Leave-sgRNA-Out'].iterrows()}

def draw_fig(df_fig, suffix, loc=4):
    if suffix == 'auc':
        lb = 'Averaged ROC-AUC'
        dt = dt_auc
    elif suffix == 'pr':
        lb = 'Averaged PR-AUC'
        dt = dt_pr
    else:
        lb = 'Averaged Spearman Correlation'
        dt = dt_r
    nx = len(xlst)
    idx = np.arange(nx) * 1.8

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect=5)

    bw = 0.15
    for i, s in enumerate(hlst):
        ys = df_fig[df_fig['h']==s]['y']
        s1 = f'{s} ({dt[s]})'
        rec = ax.bar(idx+i*bw, ys, bw, label=s, color=corlst[i])
    ax.set_ylim(0, 1)
    if lb == 'Averaged Spearman Correlation':
        ax.set_ylim(0, 0.5)
    ax.set_xlabel('Testing Strategy')
    ax.set_ylabel(lb)
    ax.xaxis.set_major_locator(ticker.FixedLocator((idx+bw/2*(len(hlst)-1))))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter((xlst)))
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(7)

    ax.legend(loc=loc, prop={'size':7})
    fig.tight_layout()
    fig.savefig(f'result/overall_{suffix}.pdf')

draw_fig(df_auc_fig, 'auc')
draw_fig(df_pr_fig, 'pr', loc=1)
draw_fig(df_r_fig, 'spearman', loc=1)










