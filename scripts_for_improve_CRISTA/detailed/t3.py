import pandas as pd
import sys
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score, average_precision_score

dfsg = joblib.load('sgdf.pkl')
dfot = joblib.load('sgotdf.pkl')


sgid = dfsg['sgid'][int(sys.argv[1])]

nm = f'data/{sgid}.csv'

df = pd.read_csv(nm, sep=',', index_col=None)

presgs = df['aligned sgRNA'].tolist()
preots = df['aligned site'].tolist()
preds = df['CRISTA score'].tolist()

sgs = list(map(lambda s: s.replace('-', '')[:-3], presgs))
assert len(set(sgs)) == 1
sg = sgs[0]
ots = list(map(lambda s: s.replace('-', ''), preots))
true_ot_set = set(dfot[dfot['sg'] == sg]['ot'].tolist())

scores = []
cur_ots = []
for ot in ots:
    if ot in true_ot_set:
        scores.append(1)
        cur_ots.append(ot)
    else:
        scores.append(0)

missing_ot = [(l, len(l)) for l in list(true_ot_set - set(cur_ots))]

print(f'id: {sgid}')
print(f'missing: {missing_ot}')
print(f'AUC: {roc_auc_score(scores, preds)}')
print(f'PR: {average_precision_score(scores, preds)}')











