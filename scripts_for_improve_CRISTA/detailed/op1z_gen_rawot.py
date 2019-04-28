import pandas as pd

acgt = [('A', '1'), ('C', '2'), ('G', '3'), ('T', '4'), ('N', '0')]


def nucrep(s):
    for k in acgt:
        s = s.replace(*k)
    return int(s)


dfot = pd.read_csv('ori_data/sgrna_ot.csv', sep=',', index_col=None)
col = dfot.columns

dfot[col[1]] = list(map(lambda s: s[:-3], dfot[col[1]]))

acgt_idx = [7, 8, 9, 22, 23, 24, 25, 26, 27, 28, 29, 30, 38, 39, 40, 41, 42]

for idx in acgt_idx:
    dfot[col[idx]] = list(map(nucrep, dfot[col[idx]]))

dfot.to_csv('data/sg_ot.csv', sep='\t', index=None)