import re
import pickle
from CRISTA import align_sequences, get_features, predict_crista_score
from sklearn.ensemble import RandomForestRegressor

sgRNA_seq = 'GAGTCCGAGCAGAAGAAGAA'
full_dna_seq = 'CCTGAGTCCGAGCAGAAGAAGAAGGGCTC'

sgRNA_seq_re = re.search("[acgtu]+", sgRNA_seq, re.IGNORECASE)
full_dna_seq_re = re.search("[acgtu]+", full_dna_seq, re.IGNORECASE)

sgRNA_seq = sgRNA_seq.upper() + "NGG"
full_dna_seq = full_dna_seq.upper()

print("Running CRISTA")
### align_sequences
aligned_sgRNA, aligned_offtarget, max_score = align_sequences(sgRNA=sgRNA_seq, genomic_extended=full_dna_seq)

### get features
features = get_features(full_dna_seq=full_dna_seq, aligned_sgRNA=aligned_sgRNA, aligned_offtarget=aligned_offtarget,
                        pa_score=max_score)
### predict

path = 'CRISTA_predictors.pkl'
with open(path, "rb") as pklr:
    predictors = pickle.load(pklr)
predictors = predictors[0]

prediction = predict_crista_score(features)
print("CRISTA predicted score:", prediction[0])
