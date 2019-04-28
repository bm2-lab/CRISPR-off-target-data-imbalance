from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr
import numpy as np

suffix = 'ns_sp'

sgids = joblib.load('sgids.pkl')
dt = joblib.load(f'result/reg/sg_div_{suffix}_res.pkl')

auc_lst = []
pr_lst = []
r_lst = []
with open(f'result1/sg_div_{suffix}_res.tab', 'w') as f:
    for sgid in sgids:
        yte, ypred = dt[sgid]
        r = spearmanr(yte, ypred)[0]
        r_lst.append(r)
        yte[yte>0] = 1
        auc = roc_auc_score(yte, ypred)
        pr = average_precision_score(yte, ypred)
        auc_lst.append(auc)
        pr_lst.append(pr)
        f.write(f'{sgid}\t{auc}\t{pr}\t{r}\n')
    auc_m = np.mean(auc_lst)
    pr_m = np.mean(pr_lst)
    r_m = np.mean(r_lst)
    f.write(f'Mean\t{auc_m}\t{pr_m}\t{r_m}\n')




