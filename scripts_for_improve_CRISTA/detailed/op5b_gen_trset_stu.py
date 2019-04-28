from sklearn.externals import joblib
import numpy as np


def rsgcomb_common(rsg):
    return list(map(lambda s: np.vstack(s), bs_common_dt[rsg]))

def rsgcomb_unique(rsg):
    return list(map(lambda s: np.vstack(s), bs_unique_dt[rsg]))

stulst, studt = joblib.load('stuids.pkl')
studies = joblib.load('studis.pkl')
stuids = list(studies.keys())
bs_common_dt = joblib.load('data/stu_common_bs_div.pkl')
common_dt = joblib.load('data/stu_common_div.pkl')
bs_unique_dt = joblib.load('data/stu_unique_bs_div.pkl')
unique_dt = joblib.load('data/stu_unique_div.pkl')

dt_common = dict.fromkeys(studies.keys())
dt_unique = dict.fromkeys(studies.keys())

for study in stuids:
    print(study)
    rest_stu = list(set(stuids) - set([study]))
    train_ds_common = list(map(lambda s: np.vstack(s), zip(*map(rsgcomb_common, rest_stu))))
    testing_ds_common = np.vstack(common_dt[study])
    xtr_lst_common = list(map(lambda s: s[:,:-1], train_ds_common))
    ytr_lst_common = list(map(lambda s: s[:,-1], train_ds_common))
    xte_common = testing_ds_common[:,:-1]
    yte_common = testing_ds_common[:,-1]
    dt_common[study] = (xtr_lst_common, ytr_lst_common, xte_common, yte_common)
    
    train_ds_unique = list(map(lambda s: np.vstack(s), zip(*map(rsgcomb_unique, rest_stu))))
    testing_ds_unique = np.vstack(unique_dt[study])
    xtr_lst_unique = list(map(lambda s: s[:,:-1], train_ds_unique))
    ytr_lst_unique = list(map(lambda s: s[:,-1], train_ds_unique))
    xte_unique = testing_ds_unique[:,:-1]
    yte_unique = testing_ds_unique[:,-1]
    dt_unique[study] = (xtr_lst_unique, ytr_lst_unique, xte_unique, yte_unique)

joblib.dump(dt_common, 'data/stu_common_div_tr.pkl', compress=3)
joblib.dump(dt_unique, 'data/stu_unique_div_tr.pkl', compress=3)
