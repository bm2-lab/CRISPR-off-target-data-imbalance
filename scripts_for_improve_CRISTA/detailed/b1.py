import pickle
from sklearn.ensemble import RandomForestRegressor

path = "CRISTA_predictors.pkl"

with open(path, "rb") as pklr:
    predictors = pickle.load(pklr)
predictors = predictors[0]

rf = RandomForestRegressor(**predictors.get_params())

