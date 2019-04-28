import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

stu_common = joblib.load('data/stu_common_div_tr_ns.pkl')
stu_unique = joblib.load('data/stu_unique_div_tr_ns.pkl')
params = joblib.load('params.pkl')
del params['criterion']
dt_common = dict.fromkeys(stu_common.keys())
dt_unique = dict.fromkeys(stu_unique.keys())

for i, study in enumerate(stu_common):
    xtr, ytr, xte, yte = stu_common[study]
    print(f'{study} ({i+1} / {len(stu_common)})')
    ytr[ytr > 0] = 1
    rf = RandomForestClassifier(**params)
    rf.fit(xtr, ytr)
    ypred = rf.predict(xte)
    dt_common[study] = (yte, ypred)
joblib.dump(dt_common, 'result/stu_common_ns_div_cls_res.pkl')

for i, study in enumerate(stu_unique):
    xtr, ytr, xte, yte = stu_unique[study]
    print(f'{study} ({i+1} / {len(stu_unique)})')
    ytr[ytr > 0] = 1
    rf = RandomForestClassifier(**params)
    rf.fit(xtr, ytr)
    ypred = rf.predict(xte)
    dt_unique[study] = (yte, ypred)
joblib.dump(dt_unique, 'result/stu_unique_ns_div_cls_res.pkl')
