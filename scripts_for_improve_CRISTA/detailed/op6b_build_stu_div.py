import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib

stu_common = joblib.load('data/stu_common_div_tr.pkl')
stu_unique = joblib.load('data/stu_unique_div_tr.pkl')
params = joblib.load('params.pkl')

dt_common = dict.fromkeys(stu_common.keys())
dt_unique = dict.fromkeys(stu_unique.keys())

for study in stu_common:
    xtr_lst, ytr_lst, xte, yte = stu_common[study]
    ypred_lst = []
    print(f'{study} begins:')
    for i in range(100):
        print(f'{study}: {i+1} / 100')
        xtr = xtr_lst[i]
        ytr = ytr_lst[i]
        rf = RandomForestRegressor(**params)
        rf.fit(xtr, ytr)
        ypred = rf.predict(xte)
        ypred_lst.append(ypred)
    ypred_m = np.mean(np.vstack(ypred_lst), axis=0)
    dt_common[study] = (yte, ypred_m)
joblib.dump(dt_common, 'result/stu_common_div_res.pkl')

for study in stu_unique:
    xtr_lst, ytr_lst, xte, yte = stu_unique[study]
    ypred_lst = []
    print(f'{study} begins:')
    for i in range(100):
        print(f'{study}: {i+1} / 100')
        xtr = xtr_lst[i]
        ytr = ytr_lst[i]
        rf = RandomForestRegressor(**params)
        rf.fit(xtr, ytr)
        ypred = rf.predict(xte)
        ypred_lst.append(ypred)
    ypred_m = np.mean(np.vstack(ypred_lst), axis=0)
    dt_unique[study] = (yte, ypred_m)

joblib.dump(dt_unique, 'result/stu_unique_div_res.pkl')
