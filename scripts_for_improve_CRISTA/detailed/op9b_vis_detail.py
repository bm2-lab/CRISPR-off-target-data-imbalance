import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('ticks')

df = pd.read_csv('result/all_detailed_res.csv', index_col=None)

fig = plt.figure()
ax = fig.add_subplot(111)

hlst = ['RF with balanced sampling', 'CRISTA', 'RF without balanced sampling', 'CFD', 'OptCD', 'CCTop']
corlst = ['blue', 'lime', 'red', 'magenta', 'gold', 'green']
par = {k: v for k, v in zip(hlst, corlst)}


def draw_box_plot(df, suffix, loc=4):
    if suffix == 'auc':
        lb = 'ROC-AUC'
    elif suffix == 'pr':
        lb = 'PR-AUC'
    elif suffix == 'spearman':
        lb = 'Spearman Correlation'
    else:
        raise ValueError('Invalid Suffix')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.boxplot(x='Strategy', y=suffix, hue='Model', ax=ax,
                data=df, palette=par)
    ax.set_ylim(0, 1)
    if lb == 'Spearman Correlation':
        ax.set_ylim(0, 0.5)
    ax.set_xlabel('Testing Strategy')
    ax.set_ylabel(lb)
    ax.legend(loc=loc, prop={'size': 7})
    fig.tight_layout()
    fig.savefig(f'result/detailed_{suffix}.pdf')


draw_box_plot(df, 'auc')
draw_box_plot(df, 'pr', loc=1)
draw_box_plot(df, 'spearman', loc=1)
