import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib


def read_data():
    pass


xtr, ytr = read_data()
params = joblib.load('params.pkl')
n_neg = len(ytr[np.isclose(ytr, 0)])
n_ot = len(ytr) - n_neg
ot_weight = n_neg / n_ot
sw = np.ones_like(ytr)
sw[~np.isclose(ytr, 0)] = ot_weight
rf = RandomForestRegressor(**params)
rf.fit(xtr, ytr, sample_weight=sw)
