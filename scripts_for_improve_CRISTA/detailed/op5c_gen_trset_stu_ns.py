from sklearn.externals import joblib
import numpy as np


def rsgcomb_common(rsg):
    return list(map(lambda s: np.vstack(s), common_dt[rsg]))

def rsgcomb_unique(rsg):
    return list(map(lambda s: np.vstack(s), unique_dt[rsg]))

stulst, studt = joblib.load('stuids.pkl')
studies = joblib.load('studis.pkl')
stuids = list(studies.keys())
common_dt = joblib.load('data/stu_common_div.pkl')
unique_dt = joblib.load('data/stu_unique_div.pkl')

dt_common = dict.fromkeys(studies.keys())
dt_unique = dict.fromkeys(studies.keys())

for study in stuids:
    print(study)
    rest_stu = list(set(stuids) - set([study]))
    train_ds_common = np.vstack(list(map(lambda s: np.vstack(s), zip(*map(rsgcomb_common, rest_stu)))))
    testing_ds_common = np.vstack(common_dt[study])
    xtr_common = train_ds_common[:,:-1]
    ytr_common = train_ds_common[:,-1]
    xte_common = testing_ds_common[:,:-1]
    yte_common = testing_ds_common[:,-1]
    dt_common[study] = (xtr_common, ytr_common, xte_common, yte_common)

    train_ds_unique = np.vstack(list(map(lambda s: np.vstack(s), zip(*map(rsgcomb_unique, rest_stu)))))
    testing_ds_unique = np.vstack(unique_dt[study])
    xtr_unique = train_ds_unique[:, :-1]
    ytr_unique = train_ds_unique[:, -1]
    xte_unique = testing_ds_unique[:, :-1]
    yte_unique = testing_ds_unique[:, -1]
    dt_unique[study] = (xtr_unique, ytr_unique, xte_unique, yte_unique)

joblib.dump(dt_common, 'data/stu_common_div_tr_ns.pkl', compress=3)
joblib.dump(dt_unique, 'data/stu_unique_div_tr_ns.pkl', compress=3)
