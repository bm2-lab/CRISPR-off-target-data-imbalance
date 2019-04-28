import pandas as pd
import numpy as np
from sklearn.externals import joblib
import os
import pyfaidx
from pyfaidx import Fasta
from CRISTA import align_sequences, get_features

faref = Fasta('{0}/hg19.fa'.format(os.environ['FASTADB']), as_raw=True, sequence_always_upper=True)


def complement_seq(seq):
    return pyfaidx.complement(seq)


def getot(row):
    chrom = 'chr{0}'.format(row['chromosome'])
    start = row['start position']
    stop = row['end position']
    strand = row['strand']
    if strand == '+':
        ot = faref[chrom][stop - 26:stop + 3]
    elif strand == '-':
        ot = complement_seq(faref[chrom][start - 4:start + 25])[::-1]
    else:
        raise ValueError('Strand not match')
    return ot


def get_seqfeatures(sg, ot):
    sg = sg.upper()
    ot = ot.upper()
    aligned_sgRNA, aligned_offtarget, max_score = align_sequences(sgRNA=sg, genomic_extended=ot)

    features = get_features(full_dna_seq=ot, aligned_sgRNA=aligned_sgRNA, aligned_offtarget=aligned_offtarget,
                            pa_score=max_score)
    return features.astype(np.float32)


def row_filt(row_raw):
    row = row_raw[1]
    chrom = row['chromosome']
    if str.isdigit(chrom) or chrom in ['X', 'Y']:
        return True
    else:
        return False


global N
global ID


def row_proc(row_raw):
    idx, row = row_raw
    sg = row['aligned sgRNA']
    ot_f = getot(row)
    features = get_seqfeatures(sg, ot_f).ravel().tolist()
    for i, fe in enumerate(features):
        row[6 + i] = fe
    print(f'{ID}: {idx+1} / {N}')
    return row


sgids = joblib.load('sgids.pkl')

for i, sgid in enumerate(sgids):
    df = pd.read_csv(f'ori_data/sg/{sgid}.csv', sep=',', index_col=None)
    col = df.columns
    df = df[col[1:-1]]
    df[col[5]] = list(map(lambda s: s.replace('-', ''), df[col[5]]))
    df[col[6]] = list(map(lambda s: s.replace('-', ''), df[col[6]]))
    ID = f'({i+1} / {len(sgids)}) {sgid}'
    N = len(df)
    df1 = pd.DataFrame(list(map(row_proc, filter(row_filt, df.iterrows()))))
    df1.to_csv(f'data/sg/{sgid}_sg.csv', sep='\t', index=None)
