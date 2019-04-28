import pandas as pd
import numpy as np
from sklearn.externals import joblib


sgids = joblib.load('sgids.pkl')
dfot_all = pd.read_csv('data/sgot.csv', sep='\t', index_col=None)

data_dt = {}

# sgid = sgids[0]
for sgid in sgids:
    print(sgid)
    dfot = dfot_all[dfot_all['Target name']==sgid]
    dfneg = pd.read_csv(f'data/neg/{sgid}_neg.csv', sep='\t', index_col=None)

    dfot_x = dfot[dfot.columns[8:-1]]
    dfot_y = dfot[dfot.columns[-1]]
    dfneg_x = dfneg[dfneg.columns[2:-1]]
    dfneg_y = dfneg[dfneg.columns[-1]]

    xot = np.array(dfot_x.to_dense(), dtype=np.float32)
    yot = np.array(dfot_y.to_dense(), dtype=np.float32)
    xneg = np.array(dfneg_x.to_dense(), dtype=np.float32)
    yneg = np.array(dfneg_y.to_dense(), dtype=np.float32)

    ot = np.hstack([xot, yot[None].T])
    neg = np.hstack([xneg, yneg[None].T])
    data_dt[sgid] = (ot, neg)

joblib.dump(data_dt, 'data/sg_div.pkl')