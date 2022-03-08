import torch.nn as nn


from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import make_pipeline,Pipeline

from sklearn.preprocessing import MinMaxScaler#,StandardScaler, FunctionTransformer, PolynomialFeatures, OneHotEncoder, PowerTransformer,
from sklearn.model_selection import cross_validate, train_test_split, RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import TransformedTargetRegressor
import matplotlib.pyplot as plt
from vb_transformers import shrinkBigKTransformer,logminus_T,exp_T,logminplus1_T,none_T,log_T, logp1_T,dropConst,columnBestTransformer
from missing_val_transformer import missingValHandler

from vb_estimators import myLogger


class TorchNet(BaseEstimator,RegressorMixin,myLogger): 
    # pytorch implementation inspired by: 
    # https://docs.microsoft.com/en-us/archive/msdn-magazine/2019/march/test-run-neural-regression-using-pytorch
    def __init__(self,do_prep=True,prep_dict={'impute_strategy':'impute_knn5'},
                 epochs=100,batch_size=32,cat_idx=None,float_idx=None,inner_cv=None):
        myLogger.__init__(self,name='torchnet.log')
        self.logger.info('starting torchnet logger')
        self.do_prep=do_prep
        self.inner_cv=inner_cv
        self.groupcount=groupcount #not used, can be deleted
        self.bestT=bestT
        self.cat_idx=cat_idx
        self.float_idx=float_idx
        self.prep_dict=prep_dict
        self.epochs=epochs
        self.batch_size=batch_size
        self.fit_counter=0 # to allow refitting/partial_fit with more data...eventually
        #self.pipe_=self.get_pipe() #formerly inside basehelper         
        #BaseHelper.__init__(self) #TODO: create basehelper for pytorch neural nets
    
    def fit(self,X,y=None,w=None):
        self.n=y.size
        max_batches=self.n//self.batch_size
        if max_batches < self.epochs:
            self.nn_n_reps=-(-self.epochs//max_batches) #ceil divide
        else:
            self.nn_n_reps=1
        self.nn_n_splits=-(-(self.n/self.nn_n_reps)//self.batch_size)
        assert not y is None
        try: 
            self.pipe_
            self.fit_counter+=1
        except:
            assert self.fit_counter==0
            self.pipe_=self.get_pipe()
        if not type(X) is 
        
    
    def get_pipe(self,):
        if self.inner_cv is None:
            n_splits=
            reps=-(-self.epochs//self.
            inner_cv=RepeatedKFold(n_splits=10, n_repeats=reps, random_state=self.fit_counter)
        else:
            inner_cv=self.inner_cv
        
        steps=[('scaler',MinMaxScaler()),('reg',TorchRegressor)]
        outerpipe=Pipeline(steps=steps)
        
        if self.do_prep:
            steps=[('prep',missingValHandler(prep_dict=self.prep_dict)),
                   ('post',outerpipe)]
            outerpipe=Pipeline(steps=steps)
        return outerpipe