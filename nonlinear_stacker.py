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
from vb_transformers import logp1_T

class stackNonLinearTransforms(BaseEstimator,TransformerMixin):
    def __init__(self,transform_list=[np.exp,logp1_T],select_best=1):
        self.transform_list=transform_list
    def fit(self,X,y=None):
        if type(X)!=pd.DataFrame:
            X=pd.DataFrame(X)
        self.X_dtypes_=dict(X.dtypes)
        self.obj_idx_=[i for i,(var,dtype) in enumerate(self.X_dtypes_.items()) if dtype=='object']
        self.float_idx_=[i for i in range(X.shape[1]) if i not in self.obj_idx_]
        self.cat_list=[X.iloc[:,idx].unique() for idx in self.obj_idx_]
        return self
    def transform(self,X,y):
        transform_tup_list=self.buildtransformers
        if type(X)!=pd.DataFrame:
            X=pd.DataFrame(X)
        categorical_T=('no_transform',none_T(),self.obj_idx_)
        if select_best:
            numeric_T=('feature_union_transformer',make_pipeline([FeatureUnion(transform_tup_list),selectKbest(score_func=f_regression)]),self.float_idx_)
        else:
            numeric_T=('feature_union_transformer',FeatureUnion(transform_tup_list),self.float_idx_)
        
        
        T=ColumnTransformer(transformers=[numeric_T,categorical_T])
        T.fit(X,y)
        X=T.transform(X)
        
    def build_transformers(self,transform_list):
        transformer_tups=[]
        for item in transform_list:
            if type(item) is np.ufunc:
                transformer_tups.append((item.__name__,FunctionTransfomer(item)))
            if type(item) is str:
                assert False, 'not developed'
            else:
                tansformer_tups.append(item.__name__,item)
        return transformer_tups
                