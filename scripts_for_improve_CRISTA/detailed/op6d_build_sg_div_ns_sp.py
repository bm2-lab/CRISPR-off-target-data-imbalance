import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib

sgids = joblib.load('sgids.pkl')
sgdiv = joblib.load('data/sg_div_tr_ns.pkl')
params = joblib.load('params.pkl')

dt = dict.fromkeys(sgids)

sgid = sgids[0]
for i, sgid in enumerate(sgids):
    print(f'{sgid} ({i+1} / {len(sgids)})')
    xtr, ytr, xte, yte = sgdiv[sgid]
    n_neg = len(ytr[np.isclose(ytr, 0)])
    n_ot = len(ytr) - n_neg
    ot_weight = n_neg / n_ot
    sw = np.ones_like(ytr)
    sw[~np.isclose(ytr, 0)] = ot_weight
    rf = RandomForestRegressor(**params)
    rf.fit(xtr, ytr, sample_weight=sw)
    ypred = rf.predict(xte)
    dt[sgid] = (yte, ypred)
joblib.dump(dt, 'result/sg_div_ns_sp_res.pkl')

