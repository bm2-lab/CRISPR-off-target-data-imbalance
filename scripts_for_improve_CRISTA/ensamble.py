import numpy as np
import sklearn
from sklearn.cross_validation import StratifiedKFold
from elevation.stacker import *

class model_ensamble:
    def __init__(self,n_model):
        self.n_model=n_model
    def fit(self,Xtrain_list,y_list):
        self.model_all=[]
        normX = True
        strength = 1.0
        for i in range(self.n_model):
            num_fold=10
            X_train=Xtrain_list[i]
            y_train=y_list[i]
            kfold_ensamble=StratifiedKFold(y_train.flatten()==0, num_fold, random_state=learn_options['seed'])
            clf_ensamble = sklearn.linear_model.LassoCV(cv=kfold_ensamble, fit_intercept=True, normalize=(~normX),n_jobs=num_fold, random_state=learn_options['seed'])
            clf_ensamble = sklearn.pipeline.Pipeline([['scaling', sklearn.preprocessing.StandardScaler()], ['lasso', clf_ensamble]])
            y_train=(y_train-np.min(y_train))/(np.max(y_train)-np.min(y_train))
            y_train=st.boxcox(y_train-y_train.min()+0.001)[0]
            self.model_i=clf_ensamble.fit(X_train,y_train)
            self.model_all.append(self.model_i)
    def predict(self,Xtest):
        prediction=[]
        for model in self.model_all:
            prediction.append(model.predict(Xtest))
        last=np.sum(i for i in prediction)/len(prediction)
        return last


def elevation_model():
    pass

