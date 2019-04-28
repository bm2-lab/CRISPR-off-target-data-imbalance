import pandas as pd
import numpy as np
from sklearn.externals import joblib


stulst, studt = joblib.load('stuids.pkl')
dfot_all = pd.read_csv('data/stuot.csv', sep='\t', index_col=None)

data_dt = {}

for stuid in stulst:
    print(stuid)
    dfot = dfot_all[dfot_all['Target name']==stuid]
    dfneg = pd.read_csv(f'data/stuneg/{stuid}_neg.csv', sep='\t', index_col=None)

    dfot_x = dfot[dfot.columns[8:-2]]
    dfot_y = dfot[dfot.columns[-2]]
    dfneg_x = dfneg[dfneg.columns[2:-1]]
    dfneg_y = dfneg[dfneg.columns[-1]]

    xot = np.array(dfot_x.to_dense(), dtype=np.float32)
    yot = np.array(dfot_y.to_dense(), dtype=np.float32)
    xneg = np.array(dfneg_x.to_dense(), dtype=np.float32)
    yneg = np.array(dfneg_y.to_dense(), dtype=np.float32)

    ot = np.hstack([xot, yot[None].T])
    neg = np.hstack([xneg, yneg[None].T])
    data_dt[stuid] = (ot, neg)

joblib.dump(data_dt, 'data/stu_sg_div.pkl')