# Data imbalanced in CRISPR off-target prediction
#genome-wide_off_target_prediction

The scripts in genome-wide_off_target_prediction are used to perform the "RF with balanced sampling" and "RF without balanced sampling" tests for genome-wide off-target profile prediction of CRISTA model.The original testing data can be referred in "Shiran Abadi et al., A machine learning approach for predicting CRISPR-Cas9 cleavage efficiencies and patterns underlying its mechanism of action. PLoS Comput Biology. 2017;10(13):e1005807."

#computational-techniques_for_data-imbanlance

The scripts in computational-techniques_for_data-imbanlance are deal with the imbanlanced data of CRISPR off-targets using two different computational techniques.  The code in ensamble.py is an ensemble learning way for the final prediction based on combning 831 trained models of original Elevation model. the code in smote.py is the SMOTE algorithm we using to process the unbanlance data before trainning the Elevation model.
