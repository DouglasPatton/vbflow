import logging, logging.handlers
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, LinearRegression, Lars
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,FunctionTransformer,StandardScaler
from sklearn.pipeline import make_pipeline,FeatureUnion,Pipeline
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.feature_selection import f_regression



class featureNameExtractor:

    """
    based on https://johaupt.github.io/scikit-learn/tutorial/python/data%20processing/ml%20pipeline/model%20interpretation/columnTransformer_feature_names.html
    Get feature names from all transformers.
    Returns
    -------
    feature_names : list of strings
        Names of the features produced by transform.
    """

    def __init__(self,column_transformer,input_features=None):
        self.column_transformer=column_transformer
        self.input_features=input_features
        self.feature_names=[]

    def run(self):

        # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
        if type(self.column_transformer) == Pipeline:
            l_transformers = [(name, trans, None, None) for step, name, trans in self.column_transformer._iter()]
        else:
            # For column transformers, follow the original method
            l_transformers = list(self.column_transformer._iter(fitted=True))


        for name, trans, columns, _ in l_transformers:
            print(f'name:{name},columns:{columns}')
            if type(trans) == Pipeline:
                # Recursive call on pipeline
                _names = featureNameExtractor(trans,input_features=self.input_features).run()
                # if pipeline has no transformer that returns names
                if len(_names)==0:
                    _names = [name + "__" + f for f in columns]
                self.feature_names.extend(_names)
            else:
                self.feature_names.extend(self.get_names(name,trans,columns))

        return self.feature_names
        
    def get_names(self,name,trans,columns):
        # >> Original get_feature_names() method
        if trans == 'drop' or (
                hasattr(columns, '__len__') and not len(columns)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                if ((not isinstance(columns, slice))
                        and all(isinstance(col, str) for col in columns)):
                    return columns
                else:
                    return column_transformer._df_columns[columns]
            else:
                indices = np.arange(column_transformer._n_features)
                return ['x%d' % i for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):
        # >>> Change: Return input column names if no method avaiable
            # Turn error into a warning
            print("Transformer %s (type %s) does not "
                                 "provide get_feature_names. "
                                 "Will return input column names if available"
                                 % (str(name), type(trans).__name__))
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            if column is None:
                return []
            else:
                return [f"{name}__{f}" for f in column]

        return [f"{name}__{f}" for f in trans.get_feature_names(input_features=self.input_features)]
    
    ### Start of processing
    


class columnBestTransformer(BaseEstimator,TransformerMixin):
    def __init__(self,float_k=None):
        self.logger=logging.getLogger()
        self.transform_funcs={
            'abs_ln':lambda x:np.log(np.abs(x)+.0000001),
            'exp':lambda x:np.exp(x/100),
            'recip':lambda x:(x+.000000001)**-1,
            'none':lambda x:x,
            'exp_inv':lambda x:np.exp(x)**-1
        }
        self.float_k=float_k
            
            
    def fit(self,X,y=None):
        if not self.float_k is None:
            Xn=X[:,:self.float_k]
        else:
            Xn=X
        self.k_=Xn.shape[1]
        #pvals=[f_regression(fn(X),y)[1][None,:] for fn in self.transform_funcs.values()]
        pvals=[]
        self.logger.info(f'Xn.shape:{Xn.shape},Xn:{Xn}')
        for fn in self.transform_funcs.values():
           
            #self.logger.info(f'fn:{fn}')
            
            TXn=fn(Xn)
            try:
                F,p=f_regression(TXn,y)
            except:
                self.logger.exception(f'error doing f_regression')
                p=np.array([10000.]*TXn.shape[1])
            pvals.append(p[None,:])
             
        pval_stack=np.concatenate(pvals,axis=0) #each row is a transform
        #self.logger.info(f'pval_stack:{pval_stack}')
        bestTloc=np.argsort(pval_stack,axis=0)[0,:]
        Ts=list(self.transform_funcs.keys())
        self.bestTlist=[Ts[i] for i in bestTloc]
        self.logger.info(f'bestTlist:{self.bestTlist},')
        T_s=list(self.transform_funcs.keys())
        self.best_T_=[T_s[loc] for loc in bestTloc]
        return self
    def transform(self,X):
        for c,t in enumerate(self.best_T_):
            X[:,c]=self.transform_funcs[t](X[:,c])
        return X
        
        
                
                

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
    def get_feature_name(self,input_features=None):
        if input_features is None:
            input_features=[f'var_{i}' for i in range(len(self.unique_))]
        return [input_features[i] for i,count in enumerate(self.unique_) if count >1]
  
    
        
                
class shrinkBigKTransformer(BaseEstimator,TransformerMixin):
    def __init__(self,max_k=None,k_share=None,selector=None):
        self.logger=logging.getLogger()
        self.max_k=max_k
        self.k_share=k_share
        self.selector=selector
            
    def get_feature_name(self,input_features=None):
        if input_features is None:
            input_features=[f'var_{i}' for i in range(len(self.k_))]
        return [input_features[i] for i in self.col_select_]
      
    def fit(self,X,y):
        assert not y is None,f'y:{y}'
        k=X.shape[1]
        self.k_=k
        if self.max_k is None:
            if self.k_share is None:
                self.max_k=k+1
            else:
                self.max_k=int(k*self.k_share)
        steps=[('scaler',StandardScaler())]
        if self.selector is None:
            self.selector='Lars'
        if self.selector=='Lars':
            steps.append(('selector',Lars(fit_intercept=1,normalize=False,n_nonzero_coefs=self.max_k)))
        elif self.selector=='elastic-net':
            steps.append(('selector',ElasticNet(fit_intercept=True,selection='random',tol=0.01,max_iter=500,warm_start=False,random_state=0,normalize=False)))
        else:
            steps.append(('selector',self.selector))
        kshrinker=Pipeline(steps=steps)
        kshrinker.fit(X,y)
        coefs=kshrinker['selector'].coef_
        self.col_select_=np.arange(k)[np.abs(coefs)>0.0001]
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

class log_T(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        XT=np.zeros(X.shape)
        XT[X>0]=np.log(X[X>0])
        return XT
    def inverse_transform(self,X,y=None):
        return np.exp(X)
    
    
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
    
class none_T(BaseEstimator,TransformerMixin): #type of transformer method that does nothing
    def __init__(self):
        pass
    def fit(self,X,y=None):
        """if type(X) is pd.DataFrame():
            self.columns_=X.columns.to_list()
        else:
            self.columns_=list(range(X.shape(1))"""
        self.k_=X.shape[1]
        return self
    def transform(self,X):
        return X
    def inverse_transform(self,X):
        return X  
    
    def get_feature_names(self,input_features=None): #gets and returns variable names
        if input_features is None:
            input_features=[f'var_{i}' for i in range(self.k_)]
        return input_features