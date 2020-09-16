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
from sklearn.pipeline import make_pipeline,FeatureUnion,Pipeline
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.feature_selection import SelectKBest,f_regression,mutual_info_regression
from vb_transformers import logp1_T,none_T
from vb_helper import myLogger


class stackNonLinearTransforms(BaseEstimator,TransformerMixin,myLogger):
    def __init__(self,transform_list=[np.exp,logp1_T],select_best=1,score_func=f_regression):
        myLogger.__init__(self,name='stackNonLinearTransforms.log')
        self.logger.info('starting stackNonLinearTransforms logger')
        self.transform_list=transform_list
        self.select_best=select_best
        self.score_func=score_func
    def fit(self,X,y):
        try:
            if type(X)!=pd.DataFrame:
                X=pd.DataFrame(X)
            #self.score_func_=self.get_score_func()
            self.X_dtypes_=dict(X.dtypes)
            self.obj_idx_=[i for i,(var,dtype) in enumerate(self.X_dtypes_.items()) if dtype=='object']
            self.float_idx_=[i for i in range(X.shape[1]) if i not in self.obj_idx_]
            self.cat_list=[X.iloc[:,idx].unique() for idx in self.obj_idx_]

            transform_tup_list=self.build_transformers(self.transform_list)
            categorical_T=('no_transform',none_T(),self.obj_idx_)
            if self.select_best:
                bestpipe=Pipeline(steps=[('featureunion',FeatureUnion(transform_tup_list)),('selectkbest',SelectKBest(score_func=self.score_func,k=self.select_best))])
                numeric_T=('feature_union_transformer',bestpipe,self.float_idx_)
            else:
                numeric_T=('feature_union_transformer',FeatureUnion(transform_tup_list),self.float_idx_)


            self.T_=ColumnTransformer(transformers=[numeric_T,categorical_T])
            self.T_.fit(X,y)
            return self
        except:
            self.logger.exception(f'')
            assert False, 'halt'
    def transform(self,X,y=None):
        
        if type(X)!=pd.DataFrame:
            X=pd.DataFrame(X)
        
        X=self.T_.transform(X)
    
    """def get_score_func(self,):
        if self.score_func=='f_regression':
            return f_regression
        elif self.score_func=='mutual_info_regression'
            return mutual_info_regression"""
    
    def build_transformers(self,transform_list):
        transformer_tups=[]
        for item in transform_list:
            if type(item) is np.ufunc:
                transformer_tups.append((item.__name__,FunctionTransformer(item)))
            elif type(item) is str:
                assert False, 'not developed'
            else:
                transformer_tups.append((item.__name__,item))
        return transformer_tups
                