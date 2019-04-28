import pandas as pd
from sklearn.externals import joblib

dfsg = joblib.load('sgdf.pkl')
sgids = dfsg['sgid'].tolist()

i = 0
df = pd.read_csv('ori_data/sgrna_ot.csv', sep=',', index_col=None)
df1 = df[df['Target name'] == sgids[i]]
df1_features = df1[df1.columns[8:]]