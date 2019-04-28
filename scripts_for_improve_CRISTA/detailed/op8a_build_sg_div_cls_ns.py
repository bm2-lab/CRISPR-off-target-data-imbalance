from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

sgids = joblib.load('sgids.pkl')
sgdiv = joblib.load('data/sg_div_tr_ns.pkl')
params = joblib.load('params.pkl')
del params['criterion']
dt = dict.fromkeys(sgids)

# sgid = sgids[0]

for i, sgid in enumerate(sgids):
    print(f'{sgid} ({i+1} / {len(sgids)})')
    xtr, ytr, xte, yte = sgdiv[sgid]
    ytr[ytr > 0] = 1
    rf = RandomForestClassifier(**params)
    rf.fit(xtr, ytr)
    ypred = rf.predict(xte)
    dt[sgid] = (yte, ypred)
joblib.dump(dt, 'result/sg_div_cls_ns_res.pkl')
