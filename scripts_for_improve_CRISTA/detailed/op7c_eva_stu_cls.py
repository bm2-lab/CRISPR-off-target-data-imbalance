from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score
import numpy as np

suffix = 'unique_ns_sp'

dt = joblib.load(f'result/cls/stu_{suffix}_div_cls_res.pkl')

auc_lst = []
pr_lst = []

with open(f'result/cls_eva/stu_div_cls_{suffix}.tab', 'w') as f:
    for stuid in dt:
        yte, ypred = dt[stuid]
        yte[yte>0] = 1
        auc = roc_auc_score(yte, ypred)
        pr = average_precision_score(yte, ypred)
        auc_lst.append(auc)
        pr_lst.append(pr)
        f.write(f'{stuid}\t{auc}\t{pr}\n')
    auc_m = np.mean(auc_lst)
    pr_m = np.mean(pr_lst)
    f.write(f'\nMean\t{auc_m}\t{pr_m}\n')




