import logging
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, LinearRegression, Lars
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,FunctionTransformer
from sklearn.pipeline import make_pipeline,FeatureUnion
from sklearn.impute import SimpleImputer,KNNImputer



                
                
                

class dropConst(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.logger=logging.getLogger()
        pass
    def fit(self,X,y=None):
        if type(X) is np.ndarray:
            X_df=pd.DataFrame(X)
        else:
            X_df=X
        self.unique_=X_df.apply(pd.Series.nunique)
        return self
    def transform(self,X):
        if type(X) is pd.DataFrame:
            return X.loc[:,self.unique_>1]
        else:
            return X[:,self.unique_>1]
  
    
        
                
class shrinkBigKTransformer(BaseEstimator,TransformerMixin):
    def __init__(self,max_k=500,selector=None):
        self.logger=logging.getLogger()
        self.max_k=max_k
        self.selector=selector
            
        
    def fit(self,X,y):
        assert not y is None,f'y:{y}'
        if self.selector is None:
            self.selector='Lars'
        if self.selector=='Lars':
            selector=Lars(fit_intercept=1,normalize=1,n_nonzero_coefs=self.max_k)
        elif self.selector=='elastic-net':
            selector=ElasticNet(fit_intercept=True,selection='random',tol=0.1,max_iter=500,warm_start=0)
        else:
            selector=self.selector
        k=X.shape[1]
        selector.fit(X,y)
        self.col_select_=np.arange(k)[np.abs(selector.coef_)>0.0001]
        #print(f'self.col_select_:{self.col_select_}')
        if self.col_select_.size<1:
            self.col_select_=np.arange(1)
            #print (f'selector.coef_:{selector.coef_}')
        return self
    
    def transform(self,X):
        return X[:,self.col_select_]


#from sklearn.compose import TransformedTargetRegressor

class logminplus1_T(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X,y=None):
        self.x_min_=np.min(X)
        return self
    def transform(self,X,y=None):
        return np.log(X-self.x_min_+1)
    def inverse_transform(self,X,y=None):
        return np.exp(X)-1+self.x_min_

class logp1_T(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.logger=logging.getLogger()
        pass
    def fit(self,X,y=None):
        xmin=X.min()
        if xmin<0:
            self.min_shift_=-xmin
        else:
            self.min_shift_=0
        self.logger.debug(f'logp1_T fitting with self.min_shift:{self.min_shift_}')
        return self
    
    def transform(self,X,y=None):
        X[X<-self.min_shift_]=-self.min_shift_ # added to avoid np.log(neg), really np.log(<1) b/c 0+1=1
        XT=np.log1p(X+self.min_shift_)
        #self.logger.info(f'logp1_T transforming XT nulls:{np.isnan(XT).sum()}')
        return  XT
        
    def inverse_transform(self,X,y=None):
        XiT=np.expm1(X)-self.min_shift_
        #self.logger.info(f'logp1_T inv transforming XiT nulls:{np.isnan(XiT).sum()}')
        try:infinites=XiT.size-np.isfinite(XiT).sum()
        except:self.logger.exception(f'type(XiT):{type(XiT)}')
        #self.logger.info(f'logp1_T inv transforming XiT not finite count:{infinites}')
        XiT[~np.isfinite(XiT)]=10**50
        return XiT
    
class logminus_T(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        return np.sign(X)*np.log(np.abs(X))
    def inverse_transform(self,X,y=None):
        return np.exp(X)
           
class exp_T(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X,y=None):
        pass
    def transform(self,X,y=None):
        return np.exp(X)
    def inverse_transform(self,X,y=None):
        xout=np.zeros(X.shape)
        xout[X>0]=np.log(X[X>0])
        return xout  
    
class none_T(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        return X
    def inverse_transform(self,X):
        return X  