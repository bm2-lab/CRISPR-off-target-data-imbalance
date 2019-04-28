import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib

sgids = joblib.load('sgids.pkl')
sgdiv = joblib.load('data/sg_div_tr.pkl')
params = joblib.load('params.pkl')

dt = dict.fromkeys(sgids)

for sgid in sgids:
    xtr_lst, ytr_lst, xte, yte = sgdiv[sgid]
    ypred_lst = []
    print(f'{sgid} begins:')
    for i in range(100):
        print(f'{sgid}: {i+1} / 100')
        xtr = xtr_lst[i]
        ytr = ytr_lst[i]
        rf = RandomForestRegressor(**params)
        rf.fit(xtr, ytr)
        ypred = rf.predict(xte)
        ypred_lst.append(ypred)
    ypred_m = np.mean(np.vstack(ypred_lst), axis=0)

    dt[sgid] = (yte, ypred_m)
joblib.dump(dt, 'result/sg_div_res.pkl')
