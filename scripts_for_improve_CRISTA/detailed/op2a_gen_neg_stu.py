import pandas as pd
from sklearn.externals import joblib

stulst, studt = joblib.load('stuids.pkl')

# stuid = stulst[0]
for stuid in stulst:
    print(stuid)
    df = pd.read_csv(f'data/study/{stuid}_stu.csv', sep='\t', index_col=None)
    col = df.columns
    cur_col = col[4:]
    df = df[cur_col]

    dfot = pd.read_csv('data/stuot.csv', sep='\t', index_col=None)
    true_ot = set(dfot[dfot['Target name'] == stuid]['aligned site'])
    cur_ot = df['aligned site'].tolist()
    notot = list(map(lambda s: s not in true_ot, cur_ot))
    df_neg = df[notot]
    df_neg['score'] = 0
    df_neg.to_csv(f'data/stuneg/{stuid}_neg.csv', sep='\t', index=None)
