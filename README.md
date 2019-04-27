Data imbalance in CRISPR off-target prediction
==========
## Indroduction
For genome-wide CRISPR off-target cleavage sites(OTS) prediction, an import issue is data imbalance – the number of true off-target cleavage sites recognized by whole-genome off-target 



detection techniques is much smaller than that of all possible nucleotide mismatch loci, making the training of machine leaning model very challenging.
In our work,we point out that 



# scripts_for_improve_CRISTA 
The scripts in scripts_for_improve_CRISTA are used to perform the "RF with balanced sampling" and "RF without balanced sampling" tests for genome-wide off-target profile prediction of CRISTA model.The original testing data can be referred in "Shiran Abadi et al., 'A machine learning approach for predicting CRISPR-Cas9 cleavage efficiencies and patterns underlying its mechanism of action'. PLoS Comput Biology. 2017;10(13):e1005807."

# scripts_for_improve_Elevation
The scripts in scripts_for_improve_Elevation deal with the imbalanced data of CRISPR off-targets using two different computational techniques. The code in ensemble.py is an ensemble learning strategy for the final prediction based on combining 831 trained models of original Elevation model. The code in smote.py is the SMOTE algorithm applied to process the imbalanced data before training the Elevation model. The two independace testing datasets are mentioned in Elevation reffered in "Listgarten, Jennifer et al. 'Prediction of off-target activities for the end-to-end design of CRISPR guide RNAs' Nature biomedical engineering vol. 2,1 (2018): 38-47. "
