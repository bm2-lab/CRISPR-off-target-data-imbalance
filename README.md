Data imbalance in CRISPR off-target prediction
==========
## Indroduction
For genome-wide CRISPR off-target cleavage sites(OTS) prediction, an import issue is data imbalance – the number of true off-target cleavage sites recognized by whole-genome off-target 
detection techniques is much smaller than that of all possible nucleotide mismatch loci, making the training of machine leaning model very challenging.<br>
In our work,we indicate that: <br>
* the benchmark of CRISPR off-target prediction should be properly evaluated and not overestimated by considering data imbalance issue
* data imbalance issue can be addressed to certain extent with the help of carefully designed computational techniques. Incorporating bootstrapping sampling technique or data synthetic technique can help to boost the CRISPR off-target prediction performance.
 
## scripts_for_improve_CRISTA
### Comprehensive analysis

CRISTA is a random-forest-based machine learning model that determines the propensity of a genomic site to be cleaved by a given single guide RNA (sgRNA) proposed by Shiran Abadi et al. In their work,they applied a bootstrapping sampling technique to under-sampling the majority class and oversampling the minority in the training datasets resulting in the training dataset with twice the size of the cleaved samples and an equal-sized set of uncleaved sample.Then they using a artificial dataset which contains cleaved sites plus an equal set of randomly-chosen uncleaved sites from the original dataset to evaluate their model.<br>
In our work, we do a comprehensive reanalysis based on CRISTA
* We evaluate CRISTA using the the whole testing dataset instead of its balanced codition to indicate that only the complete testing dataset can reflect the performance of a model.
* we point out that the routine training methods without bootstrapping sampling generally perform worse than methods to the contrary(denoted as “RF without balanced sampling”)
* we point out that that sampling an equal size of positive and negative samples during the training of random forest model(denoted as "RF with balanced sampling")

### Codes used for analysis 
Scripts in the catalog of scripts_for_improve_CRISTA are used to perform leave-one-sgRNA-out and leave-study-out testing experiments using the complete datasets. The methods of “RF with balanced sampling" and "RF without balanced sampling" are all used to address the data imbalanced issue.

### The citation of CRISTA
Shiran Abadi, W.X.Y., David Amar, Itay Mayrose, A machine learning approach for predicting CRISPR-Cas9 cleavage efficiencies and patterns underlying its mechanism of action. PLoS Comput Biology, 2017. 10(13): p. e1005807

## scripts_for_improve_Elevation
### Comprehensive analysis
Elevation is a two-layerd model for OTS predition. Particularly, the published GUIDE-seq data comprising nine sgRNAs assessed for off-target cleavage activity were used to train the second-layer. These sgRNAs yielded 354 active off-target sites with up to six mismatches. Inactive sites were obtained by Cas-OFFinder to identify all 294,534 sites with six or fewer mismatches as the negative sample pool. Elevation selected 355 inactive samples among all negative samples with customized interval to conduct their training process.<br>
In our work,  efficient computational techniques are used based on Elevation to improve sensitivity and specificity of OTS prediction.
* using bootstrapping sampling-based ensemble learning techniques can significantly enhance the performance of OTS prdiction machine learning model. 
* using the Synthetic Minority Over-sampling Technique (SMOTE) makes contribution to imporve the performance of OTS prediciton machine learning model.

### Code used for analysis
Scripts in the catalog of scripts_for_improve_Elevation are the computational tech



The scripts in scripts_for_improve_Elevation deal with the imbalanced data of CRISPR off-targets using two different computational techniques. The code in ensemble.py is an ensemble learning strategy for the final prediction based on combining 831 trained models of original Elevation model. The code in smote.py is the SMOTE algorithm applied to process the imbalanced data before training the Elevation model. The two independace testing datasets are mentioned in Elevation reffered in "Listgarten, Jennifer et al. 'Prediction of off-target activities for the end-to-end design of CRISPR guide RNAs' Nature biomedical engineering vol. 2,1 (2018): 38-47. "
