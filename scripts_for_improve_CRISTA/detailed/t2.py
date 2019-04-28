import pandas as pd
from sklearn.externals import joblib

dfstu = pd.read_csv('ori_data/study_ot.csv', sep=',', index_col=None)
dfot = pd.read_csv('data/sgot.csv', sep='\t', index_col=None)

dfstu['sgRNA sequence'] = list(map(lambda s: s[:-3], dfstu['sgRNA sequence']))


s1 = set(dfstu['sgRNA sequence'])
s2 = set(dfot['aligned sgRNA'])
s = s1 - s2

sg = dfstu['sgRNA sequence'].tolist()
other = list(map(lambda k: k in s, sg))
tar = dfstu[other]['Target name'].drop_duplicates()

