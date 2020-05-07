"Python code repository for "Enhanced Geometric SMOTE [EG-SMOTE]"  

EG-SMOTE algorithm is an enhancement of G-SMOTE algorithm for efficient resampling that addresses the limitations of 
* Synthesizing noisy minority samples
* Overfitting due to extreme synthesis of minority samples
* Improper synthesis along the borderlines.
  
Two types of imbalances are handled in EG-SMOTE; 
1) between-class imbalance
    - imbalance is handled with resampling methodology
2) within-class imbalance
    - rectified by applying the k-means clustering algorithm and then applying resampling within clusters. 

**Requirement Modules**   
* Python 3
* numpy
* pandas
* imblearn
* Scikit-learn  


**Getting Started**  

A sample dataset can be found in the Experiment/data/NSLKDD-mini.csv  

The model architecture can be found in egsmote.py. And you can try the demo experiment by running Experiment/demo.py file.
