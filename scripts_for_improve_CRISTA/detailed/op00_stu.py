import pandas as pd
from sklearn.externals import joblib

dfstu = pd.read_csv('ori_data/study_ot_ori.csv', index_col=None)
dfot = pd.read_csv('ori_data/sgrna_ot.csv', index_col=None)
dfot['sg'] = list(map(lambda s: s[:-3], dfot['aligned sgRNA']))
df1 = dfot[['Target name', 'sg']].drop_duplicates()
dt = {row[1]['sg']: row[1]['Target name'] for row in df1.iterrows()}
dt['TCCTCCTCCCCACCCACCTT'] = 'OT2'
dt['ACCCCTTCCCCACCTACCTT'] = 'OT1'
dt['GCCTCTCCCCACCCACCCTT'] = 'OT17'
dt['GCCTCTTTCCCACCCACCTT'] = 'RAG1A'
dt['GACTTGTTTTCATTGTTCTC'] = 'RAG1B'

dfstu['sg'] = list(map(lambda s: s[:-3], dfstu['aligned sgRNA']))
dfstu['neg'] = [dt[s] for s in dfstu['sg']]
del dfstu['sg']
dfstu['Target name'][dfstu['Target name'] == 'Chr7 103.6Mb OT17'] = 'OT17'
dfstu['Target name'][dfstu['Target name'] == 'Chr19 DAZAP1 OT1'] = 'OT1'
dfstu['Target name'][dfstu['Target name'] == 'Chr12 47.0Mb OT2'] = 'OT2'
dfstu['Study'][dfstu['Study'].str.contains('guideSeq')] = 'guideSeq'

dfstu['Target name'] = list(map(lambda k: f'{k[0]}_{k[1]}', zip(dfstu['Study'], dfstu['Target name'])))

df3 = dfstu[['Study', 'Target name', 'aligned sgRNA', 'neg']].drop_duplicates()

stulst = []
studt = {}
studies = {}
for s in df3['Study'].drop_duplicates().tolist():
    studies[s] = []

for row_raw in df3.iterrows():
    row = row_raw[1]
    stulst.append(row['Target name'])
    studt[row['Target name']] = (row['aligned sgRNA'], row['neg'])
    studies[row['Study']].append(row['Target name'])

# joblib.dump((stulst, studt), 'stuids.pkl')
# joblib.dump(studies, 'studis.pkl')
# dfstu.to_csv('ori_data/study_ot.csv', index=None)
