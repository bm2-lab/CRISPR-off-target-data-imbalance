import elevation.prediction_pipeline as pp
import elevation
import random
from sklearn.neighbors import NearestNeighbors
import numpy as np
import elevation
import pandas
import azimuth
import joblib
import logging
from joblib import Memory
from elevation.model_comparison import *
import copy
import scipy.stats as ss
from sklearn.grid_search import ParameterGrid
import sklearn.linear_model
import scipy as sp
import scipy.stats
import elevation.models
import elevation.features
#import GPy
import socket
from elevation.stacker import *
import elevation.util as ut
from sklearn.metrics import auc, roc_curve
from elevation import settings
import sklearn.isotonic
from sklearn.cross_validation import StratifiedKFold
import sklearn.pipeline
import sklearn.preprocessing
import pandas as pd
from elevation.cmds.predict import Predict
from elevation import options


def filter_pam_out_of_muts(data, i):
    tmp_muts = data['mut positions'].iloc[i]
    # because Hsu-Zhang ignores alternate PAMs which we have encoded with '22'
    pam_pos = 22
    if pam_pos in tmp_muts:
        tmp_muts.remove(pam_pos)
    tmp_muts = np.array(tmp_muts)
    num_m = len(tmp_muts)
    return num_m, tmp_muts


def predict(model, data, learn_options, learn_options_override=None, verbose=False):
    if learn_options_override is None:
        learn_options_override = learn_options
    predictions, model, learn_options, _tmpdata, feature_names, all_predictions_ind = predict_elevation(data=data,
                                                                                                        model=(model,
                                                                                                               learn_options),
                                                                                                        model_file=None,
                                                                                                        pam_audit=False,
                                                                                                        learn_options_override=learn_options_override,
                                                                                                        force_zero_intercept=False,
                                                                                                        naive_bayes_combine=True,
                                                                                                        verbose=verbose)
    return predictions, all_predictions_ind


class model_ensamble:
    def __init__(self, n_model):
        self.n_model = n_model

    def fit(self, Xtrain_list, y_list):
        self.model_all = []
        normX = True
        strength = 1.0
        for i in range(self.n_model):
            num_fold = 10
            X_train = Xtrain_list[i]
            y_train = y_list[i]
            kfold_ensamble = StratifiedKFold(y_train.flatten() == 0, num_fold, random_state=learn_options['seed'])
            clf_ensamble = sklearn.linear_model.LassoCV(cv=kfold_ensamble, fit_intercept=True, normalize=(~normX),
                                                        n_jobs=num_fold, random_state=learn_options['seed'])
            clf_ensamble = sklearn.pipeline.Pipeline(
                [['scaling', sklearn.preprocessing.StandardScaler()], ['lasso', clf_ensamble]])
            y_train = (y_train - np.min(y_train)) / (np.max(y_train) - np.min(y_train))
            y_train = st.boxcox(y_train - y_train.min() + 0.001)[0]
            self.model_i = clf_ensamble.fit(X_train, y_train)
            self.model_all.append(self.model_i)

    def predict(self, Xtest):
        prediction = []
        for model in self.model_all:
            prediction.append(model.predict(Xtest))
        last = np.sum(i for i in prediction) / len(prediction)
        return last


def stacked_predictions(data, preds_base_model,
                        models=['product', 'CFD', 'constant-power', 'linear-raw-stacker', 'linreg-stacker',
                                'RF-stacker', 'GP-stacker', 'raw GP'],
                        truth=None, guideseq_data=None, preds_guideseq=None, prob_calibration_model=None,
                        learn_options=None, return_model=False, trained_model=None,
                        models_to_calibrate=None, return_residuals=False):  # , dnase_train=None, dnase_test=None):

    predictions = dict([(m, None) for m in models])

    num_mismatches = np.array([len(t) for t in data["Annotation"].values])

    # if ('use_mut_distances' in learn_options.keys() and learn_options['use_mut_distances']):
    data = elevation.features.extract_mut_positions_stats(data)

    if guideseq_data is not None:
        y = guideseq_data['GUIDE-SEQ Reads'].values[:, None]
        num_annot = np.array([len(t) for t in guideseq_data["Annotation"].values])

    if 'logistic stacker' in models:
        X = preds_guideseq.copy()
        Xtest = preds_base_model.copy()
        m = Stacker(y, X, warp_out=False)
        m.maximize()
        predictions['logistic stacker'] = m.predict(Xtest)

    if 'CFD' in models:
        # predicting
        if 'cfd_table_file' not in learn_options.keys():
            learn_options['cfd_table_file'] = settings.pj(settings.offtarget_data_dir,
                                                          "STable 19 FractionActive_dlfc_lookup.xlsx")

        cfd = elevation.models.CFDModel(cfd_table_file=learn_options['cfd_table_file'])

        predictions['CFD'] = cfd.predict(data["Annotation"].values, learn_options["num_proc"])[:, None]

    if 'product' in models:
        predictions['product'] = np.nanprod(preds_base_model, axis=1)[:, None]

    if 'constant-power' in models:
        predictions['constant-power'] = np.power(0.5, num_mismatches)

    if 'CCTOP' in models:
        # predicting
        term1 = np.zeros((data.shape[0], 1))
        for i in range(len(term1)):
            num_m, tmp_muts = filter_pam_out_of_muts(data, i)
            term1[i] = np.sum(1.2 ** np.array(tmp_muts))
        predictions['CCTOP'] = -term1.flatten()

    if 'HsuZhang' in models:

        # predicting
        W = [0.0, 0.0, 0.014, 0.0, 0.0, 0.395, 0.317, 0, 0.389, 0.079, 0.445, 0.508, 0.613, 0.851, 0.732, 0.828, 0.615,
             0.804, 0.685, 0.583]
        pred = np.zeros((data.shape[0], 1))

        for i in range(len(pred)):
            num_m, tmp_muts = filter_pam_out_of_muts(data, i)

            if len(tmp_muts) == 0:
                pred[i] = 1.0
            else:
                d = ut.get_pairwise_distance_mudra(tmp_muts)
                term1 = np.prod(1. - np.array(W)[tmp_muts - 1])

                if num_m > 1:
                    term2 = 1. / (((19 - d) / 19) * 4 + 1)
                else:
                    term2 = 1

                term3 = 1. / (num_m) ** 2
                pred[i] = term1 * term2 * term3

        predictions['HsuZhang'] = pred.flatten()

    if 'linear-raw-stacker' in models or 'GBRT-raw-stacker' in models:

        if trained_model is None:
            # put together the training data
            X = preds_guideseq.copy()
            X[np.isnan(X)] = 1.0
            feature_names = ['pos%d' % (i + 1) for i in range(X.shape[1])]
            # adding product, num. annots and sum to log of itself
            X = np.concatenate((np.log(X), np.prod(X, axis=1)[:, None], num_annot[:, None], np.sum(X, axis=1)[:, None]),
                               axis=1)
            feature_names.extend(['product', 'num. annotations', 'sum'])
            # X = np.log(X)

            # Only product
            # X = np.prod(X, axis=1)[:, None]
            # feature_names = ['product']

        Xtest = preds_base_model.copy()
        Xtest[np.isnan(Xtest)] = 1.0
        Xtest = np.concatenate(
            (np.log(Xtest), np.prod(Xtest, axis=1)[:, None], num_mismatches[:, None], np.sum(Xtest, axis=1)[:, None]),
            axis=1)
        # Xtest = np.log(Xtest)
        # Xtest = np.prod(Xtest, axis=1)[:, None]

        if ('use_mut_distances' in learn_options.keys() and learn_options['use_mut_distances']):
            guideseq_data = elevation.features.extract_mut_positions_stats(guideseq_data)
            X_dist = guideseq_data[
                ['mut mean abs distance', 'mut min abs distance', 'mut max abs distance', 'mut sum abs distance',
                 'mean consecutive mut distance', 'min consecutive mut distance', 'max consecutive mut distance',
                 'sum consecutive mut distance']].values
            Xtest_dist = data[
                ['mut mean abs distance', 'mut min abs distance', 'mut max abs distance', 'mut sum abs distance',
                 'mean consecutive mut distance', 'min consecutive mut distance', 'max consecutive mut distance',
                 'sum consecutive mut distance']].values
            X = np.concatenate((X, X_dist), axis=1)
            Xtest = np.concatenate((Xtest, Xtest_dist), axis=1)

        if 'azimuth_score_in_stacker' in learn_options.keys() and learn_options['azimuth_score_in_stacker']:
            azimuth_score = elevation.model_comparison.get_on_target_predictions(guideseq_data, ['WT'])[0]
            X = np.concatenate((X, azimuth_score[:, None]), axis=1)

            azimuth_score_test = elevation.model_comparison.get_on_target_predictions(data, ['WT'])[0]
            Xtest = np.concatenate((Xtest, azimuth_score_test[:, None]), axis=1)

        if 'linear-raw-stacker' in models:

            dnase_type = [key for key in learn_options.keys() if 'dnase' in key]
            assert len(dnase_type) <= 1
            if len(dnase_type) == 1:
                dnase_type = dnase_type[0]
                use_dnase = learn_options[dnase_type]
            else:
                use_dnase = False

            if use_dnase:

                dnase_train = guideseq_data["dnase"].values
                dnase_test = data["dnase"].values
                assert dnase_train.shape[0] == X.shape[0]
                assert dnase_test.shape[0] == Xtest.shape[0]

                if dnase_type == 'dnase:default':
                    # simple appending (Melih)
                    X = np.concatenate((X, dnase_train[:, None]), axis=1)
                    Xtest = np.concatenate((Xtest, dnase_test[:, None]), axis=1)

                elif dnase_type == 'dnase:interact':
                    # interaction with original features
                    X = np.concatenate((X, X * dnase_train[:, None]), axis=1)
                    Xtest = np.concatenate((Xtest, Xtest * dnase_test[:, None]), axis=1)

                elif dnase_type == 'dnase:only':
                    # use only the dnase
                    X = dnase_train[:, None]
                    Xtest = dnase_test[:, None]

                elif dnase_type == 'dnase:onlyperm':
                    # use only the dnase
                    pind = np.random.permutation(dnase_train.shape[0])
                    pind_test = np.random.permutation(dnase_test.shape[0])
                    X = dnase_train[pind, None]
                    Xtest = dnase_test[pind_test, None]
                else:
                    raise NotImplementedError("no such dnase type: %s" % dnase_type)

            normX = True
            strength = 1.0

            # train the model
            if trained_model is None:

                # subsample the data for more balanced training

                ind_zero = np.where(y == 0)[0]
                ind_keep = (y != 0).flatten()
                nn = ind_keep.sum()
                increment = int(ind_zero.shape[0] / float(nn))
                X_positive = X[ind_keep]  # 找出正样本和负样本
                X_negative = X[ind_keep == False]
                y_positive = y[ind_keep]
                y_negative = y[ind_keep == False]
                X_train_list = []
                y_train_list = []
                list1 = [i for i in range(len(y_negative))]
                for i in range(increment):  # 分配X_train_list 与y_train_list的比例.
                    tmp = random.sample(list1, len(X_positive))
                    xtmp = X[tmp]
                    ytmp = y[tmp]
                    X_train = np.concatenate((X_positive, xtmp))
                    y_train = np.concatenate((y_positive, ytmp))
                    X_train_list.append(X_train)
                    y_train_list.append(y_train)

                clf = model_ensamble(n_model=increment)
                clf.fit(X_train_list, y_train_list)
                predictions['linear-raw-stacker'] = clf.predict(Xtest)

            else:
                clf = trained_model

            # if normX:
            #    predictions['linear-raw-stacker'] = clf.predict(normalizeX(Xtest, strength, None))
            # else:
            predictions['linear-raw-stacker'] = clf.predict(Xtest)
            # residuals = np.log(y[ind_keep].flatten()+0.001) - clf.predict(X[ind_keep])

    if 'linreg-stacker' in models:
        m_stacker = StackerFeat()
        m_stacker.fit(preds_guideseq, y, model='linreg', normalize_feat=False)
        predictions['linreg-stacker'] = m_stacker.predict(preds_base_model)

    if 'RF-stacker' in models:
        m_stacker = StackerFeat()
        m_stacker.fit(preds_guideseq, y, model='RFR', normalize_feat=False)
        predictions['RF-stacker'] = m_stacker.predict(preds_base_model)

    if 'GP-stacker' in models:
        m_stacker = StackerFeat()
        m_stacker.fit(preds_guideseq, y, model='GP', normalize_feat=False)
        predictions['GP-stacker'] = m_stacker.predict(preds_base_model)

    if 'raw GP' in models:
        X = preds_guideseq.copy()
        X[np.isnan(X)] = 1.0
        D_base_predictions = X.shape[1]
        X = np.concatenate((np.prod(X, axis=1)[:, None],
                            num_annot[:, None],
                            np.sum(X, axis=1)[:, None],
                            X), axis=1)

        Xtest = preds_base_model.copy()
        Xtest[np.isnan(Xtest)] = 1.0
        Xtest = np.concatenate((np.prod(Xtest, axis=1)[:, None],
                                num_mismatches[:, None],
                                np.sum(Xtest, axis=1)[:, None],
                                Xtest), axis=1)

        K = GPy.kern.RBF(1, active_dims=[0]) + GPy.kern.RBF(1, active_dims=[1]) + GPy.kern.Linear(1, active_dims=[
            2]) + GPy.kern.RBF(D_base_predictions, active_dims=range(3, D_base_predictions + 3))
        m = GPy.models.GPRegression(X, np.log(y), kernel=K)
        m.optimize_restarts(5, messages=0)
        predictions['raw GP'] = m.predict(Xtest)[0]

    if 'combine' in models:
        predictions['combine'] = np.ones_like(predictions[predictions.keys()[0]])

        for c_model in models:
            if c_model != 'combine':
                predictions['combine'] += predictions[c_model].flatten()[:, None]
        predictions['combine'] /= len(models) - 1

    if 'ensemble' in models:
        predictions['ensemble'] = (predictions['product'].flatten() + predictions['linear-raw-stacker'].flatten()) / 2.

    if prob_calibration_model is not None:

        if models_to_calibrate is None:
            models_to_calibrate = ['linear-raw-stacker']

        for m in models:

            if False:  # m == 'linear-raw-stacker':
                pred = np.exp(predictions[m].flatten()[:, None]) - 0.001  # undo log transformation
            else:
                pred = predictions[m].flatten()[:, None]

            if m in models_to_calibrate:

                cal_pred = prob_calibration_model[m].predict_proba(pred)[:, 1]
                # cal_pred = prob_calibration_model[m].predict_proba(pred)[:, 0]

                if len(pred) > 10:
                    assert np.allclose(sp.stats.spearmanr(pred, cal_pred)[0],
                                       1.0)  # or np.allclose(sp.stats.spearmanr(pred, cal_pred)[0], -1.0)

                predictions[m] = cal_pred

    if truth is not None:
        res_str = "Spearman r: "
        for m in models:
            res_str += "%s=%.3f " % (m, sp.stats.spearmanr(truth, predictions[m])[0])
        print
        res_str

        res_str = "NDCG: "
        for m in models:
            res_str += "%s=%.3f " % (
            m, azimuth.metrics.ndcg_at_k_ties(truth.values.flatten(), predictions[m].flatten(), truth.shape[0]))
        print
        res_str

    if return_model:
        if return_residuals:
            return predictions, clf, feature_names, residuals
        else:
            return predictions, clf, feature_names

    return predictions


def train_prob_calibration_model(cd33_data, guideseq_data, preds_guideseq, base_model, learn_options,
                                 which_stacker_model='linear-raw-stacker', other_calibration_models=None):
    assert which_stacker_model == 'linear-raw-stacker', "only LRS can be calibrated right now"

    Y_bin = cd33_data['Day21-ETP-binarized'].values
    Y = cd33_data['Day21-ETP'].values

    nb_pred, individual_mut_pred_cd33 = predict(base_model, cd33_data, learn_options)

    # get linear-raw-stacker (or other model==which_model) predictions, including training of that model if appropriate (e.g. linear-raw-stacker)
    X_guideseq, clf_stacker_model, feature_names_stacker_model = stacked_predictions(cd33_data,
                                                                                     individual_mut_pred_cd33,
                                                                                     models=[which_stacker_model],
                                                                                     guideseq_data=guideseq_data,
                                                                                     preds_guideseq=preds_guideseq,
                                                                                     learn_options=learn_options,
                                                                                     models_to_calibrate=None,
                                                                                     prob_calibration_model=None,
                                                                                     return_model=True)
    X_guideseq = X_guideseq[which_stacker_model]

    clf = sklearn.linear_model.LogisticRegression(fit_intercept=True, solver='lbfgs')

    # fit the linear-raw-stacker (or whatever model is being calibrated) predictions on cd33 to the actual binary cd33 values
    clf.fit(X_guideseq[:, None], Y_bin)
    y_pred = clf.predict_proba(X_guideseq[:, None])[:, 1]
    # y_pred = clf.predict_proba(X_guideseq[:, None])[:, 0]

    # import ipdb; ipdb.set_trace()

    expected_sign = np.sign(sp.stats.spearmanr(X_guideseq, Y_bin)[0])
    assert np.allclose(sp.stats.spearmanr(y_pred, X_guideseq)[0], 1.0 * expected_sign, atol=1e-2)

    return clf


def excute(wildtype, offtarget, calibration_models, base_model, guideseq_data, preds_guideseq,
           learn_options):  # the function for tesing model
    start = time.time()
    wt = wildtype
    mut = offtarget
    df = pd.DataFrame(columns=['30mer', '30mer_mut', 'Annotation'], index=range(len(wt)))
    df['30mer'] = wt
    df['30mer_mut'] = mut
    annot = []
    for i in range(len(wt)):
        annot.append(elevation.load_data.annot_from_seqs(wt[i], mut[i]))
    df['Annotation'] = annot
    # print "Time spent parsing input: ", time.time() - start

    base_model_time = time.time()
    nb_pred, individual_mut_pred = elevation.prediction_pipeline.predict(base_model, df, learn_options)
    # print "Time spent in base model predict(): ", time.time() - base_model_time

    start = time.time()
    pred = stacked_predictions(df, individual_mut_pred,
                               learn_options=learn_options,
                               guideseq_data=guideseq_data,
                               preds_guideseq=preds_guideseq,
                               prob_calibration_model=calibration_models,
                               models=['HsuZhang', 'CFD', 'CCTOP', 'linear-raw-stacker'])
    return pred