from sklearn.externals import joblib
import numpy as np


def rsgcomb(rsg):
    return list(map(lambda s: np.vstack(s), sg_dt[rsg]))


sg_dt = joblib.load('data/sg_div.pkl')
sgids = joblib.load('sgids.pkl')

dt = dict.fromkeys(sgids)

for sgid in sgids:
    print(sgid)
    rest_sg = list(set(sgids) - set([sgid]))
    train_ds = np.vstack(list(map(lambda s: np.vstack(s), zip(*map(rsgcomb, rest_sg)))))
    testing_ds = np.vstack(sg_dt[sgid])
    xtr = train_ds[:,:-1]
    ytr = train_ds[:,-1]
    xte = testing_ds[:,:-1]
    yte = testing_ds[:,-1]
    dt[sgid] = (xtr, ytr, xte, yte)
joblib.dump(dt, 'data/sg_div_tr_ns.pkl', compress=3)
