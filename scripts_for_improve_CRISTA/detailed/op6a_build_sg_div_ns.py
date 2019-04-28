from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib

sgids = joblib.load('sgids.pkl')
sgdiv = joblib.load('data/sg_div_tr_ns.pkl')
params = joblib.load('params.pkl')

dt = dict.fromkeys(sgids)

# sgid = sgids[0]

for i, sgid in enumerate(sgids):
    print(f'{sgid} ({i+1} / {len(sgids)})')
    xtr, ytr, xte, yte = sgdiv[sgid]
    rf = RandomForestRegressor(**params)
    rf.fit(xtr, ytr)
    ypred = rf.predict(xte)
    dt[sgid] = (yte, ypred)
joblib.dump(dt, 'sg_div_ns_res.pkl')
