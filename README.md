# vbflow

## example notebooks
preprocessing with imputation for missing values + linear regression model selection 
 - [view notebook](https://nbviewer.jupyter.org/github/DouglasPatton/vbflow/blob/master/estimator_comparison.ipynb) 
 
 
 - Imputation
  - dropping approaches
     - drop-col-X  drop columns missing >X%, drop remaining rows with missing vals
     - drop-row drop rows with missing vals
  - middle approaches
     - numeric: mean
     - categorical: median
  - numeric variables

  - categorical variables
 
 - Variable selection
   - linear regression based
     - lars https://scikit-learn.org/stable/modules/linear_model.html#least-angle-regression
     - elastic net 
   - PCA
   - cv regularization hyper-parameter tuning
     - SVM
     - elasti-net
  - 
  
  - 
 
 
## tasks:

   - [x] handle missing values,  
   - [x] make interactions
   - [x] make transformed X columns,
   - [x] transform Y
   - [-] generalized regression estimators
           
   

   
   

   
## future steps
 - [ ] classifiers
 - [ ] add options for dealing with important but underrepresented values of Y
 - [ ] construct model averaging estimators using average cross validation scores
 - [ ] predict with confidence intervals 
  -    [ ] could use variability from cv
  -    [ ] some algorithms have their own method

   
