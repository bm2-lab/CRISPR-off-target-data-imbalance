from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib


def read_data():
    pass


xtr, ytr = read_data()
params = joblib.load('params.pkl')
rf = RandomForestRegressor(**params)
rf.fit(xtr, ytr)
