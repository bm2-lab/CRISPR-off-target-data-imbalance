import pandas as pd
from sklearn.externals import joblib


sgids = joblib.load('sgids.pkl')

# sgid = sgids[0]
for sgid in sgids:
    print(sgid)
    df = pd.read_csv(f'data/sg/{sgid}_sg.csv', sep='\t', index_col=None)
    col = df.columns
    cur_col = col[4:]
    df = df[cur_col]

    dfot = pd.read_csv('data/sgot.csv', sep='\t', index_col=None)
    true_ot = set(dfot[dfot['Target name']==sgid]['aligned site'])

    cur_ot = df['aligned site'].tolist()
    notot = list(map(lambda s: s not in true_ot, cur_ot))
    df_neg = df[notot]
    df_neg['score'] = 0
    df_neg.to_csv(f'data/neg/{sgid}_neg.csv', sep='\t', index=None)













