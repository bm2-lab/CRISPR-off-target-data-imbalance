import pandas as pd
import numpy as np
from sklearn.externals import joblib


stu_common = {'BLESS': ['BLESS_EMX1(1)', 'BLESS_VEGFA(1)'],
              'HTGTS': ['HTGTS_EMX1', 'HTGTS_VEGFA'],
              'guideSeq': ['guideSeq_EMX1', 'guideSeq_EMX1_1', 'guideSeq_FANCF', 'guideSeq_FANCF-1',
                           'guideSeq_VEGFA_site1', 'guideSeq_VEGFA_site2', 'guideSeq_VEGFA_2',
                           'guideSeq_VEGFA_site3', 'guideSeq_VEGFA_3']}

stulst, studt = joblib.load('stuids.pkl')
studies = joblib.load('studis.pkl')
data_dt = joblib.load('data/stu_sg_div.pkl')
stu_unique = {}
for s in studies:
    stu_unique[s] = []
    for g in studies[s]:
        if g not in stu_common[s]:
            stu_unique[s].append(g)

dt_common = dict.fromkeys(studies.keys())
dt_unique = dict.fromkeys(studies.keys())

def sgs_comb(sgs):
    def foo(t):
        k = list(zip(*t))
        ot = np.vstack(k[0])
        neg = np.vstack(k[1])
        return ot, neg

    ds = [data_dt[sg] for sg in sgs]
    return foo(ds)

for s in dt_common:
    sgs = stu_common[s]
    dt_common[s] = sgs_comb(sgs)

for s in dt_unique:
    sgs = stu_unique[s]
    dt_unique[s] = sgs_comb(sgs)


joblib.dump(dt_common, 'data/stu_common_div.pkl')
joblib.dump(dt_unique, 'data/stu_unique_div.pkl')





