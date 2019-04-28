from sklearn.externals import joblib
import numpy as np


def rsgcomb(rsg):
    return list(map(lambda s: np.vstack(s), bs_dt[rsg]))


sg_dt = joblib.load('data/sg_div.pkl')
sgids = joblib.load('sgids.pkl')
bs_dt = joblib.load('data/sg_bs_div.pkl')
dt = dict.fromkeys(sgids)
# sgid = sgids[0]
for sgid in sgids:
    print(sgid)
    rest_sg = list(set(sgids) - set([sgid]))
    train_ds = list(map(lambda s: np.vstack(s), zip(*map(rsgcomb, rest_sg))))
    testing_ds = np.vstack(sg_dt[sgid])
    xtr_lst = list(map(lambda s: s[:,:-1], train_ds))
    ytr_lst = list(map(lambda s: s[:,-1], train_ds))
    xte = testing_ds[:,:-1]
    yte = testing_ds[:,-1]
    dt[sgid] = (xtr_lst, ytr_lst, xte, yte)
joblib.dump(dt, 'data/sg_div_tr.pkl', compress=3)
