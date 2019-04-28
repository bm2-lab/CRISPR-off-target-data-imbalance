import pandas as pd
from sklearn.externals import joblib


acgt = [('A', '1'), ('C', '2'), ('G', '3'), ('T', '4'), ('N', '0')]


def nucrep(s):
    for k in acgt:
        s = s.replace(*k)
    return int(s)

dfsg = joblib.load('sgdf.pkl')
sgids = dfsg['sgid'].tolist()

# sgid = sgids[0]
for sgid in sgids:
    print(sgid)
    df = pd.read_csv(f'ori_data/{sgid}.csv', sep=',', index_col=None)
    col = df.columns
    df[col[5]] = list(map(lambda s: s.replace('-', '')[:-3], df[col[5]]))
    df[col[6]] = list(map(lambda s: s.replace('-', ''), df[col[6]]))

    acgt_idx = [i+4 for i in [7, 8, 9, 22, 23, 24, 25, 26, 27, 28, 29, 30, 38, 39, 40, 41, 42]]
    for idx in acgt_idx:
        df[col[idx]] = list(map(nucrep, df[col[idx]]))
    cur_col = col[5:-1]
    df = df[cur_col]

    dfot = pd.read_csv('data/sg_ot.csv', sep='\t', index_col=None)
    true_ot = set(dfot[dfot['Target name']==sgid]['DNA site sequence'])

    cur_ot = df['aligned site'].tolist()
    notot = list(map(lambda s: s not in true_ot, cur_ot))
    df_neg = df[notot]
    df_neg['score'] = 0
    df_neg.to_csv(f'data/neg/{sgid}_neg.csv', sep='\t', index=None)













