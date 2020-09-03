import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV, LinearRegression, Lars
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer,KNNImputer

from vb_helper import none_T#,VBHelper,shrinkBigKTransformer,logminus_T,exp_T,logminplus1_T,logp1_T,missingValHandler,dropConst





class missingValHandler(BaseEstimator,TransformerMixin):
    # https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html#use-columntransformer-by-selecting-column-by-names
    def __init__(self,strategy='drop_row',transformer=None):
        self.strategy=strategy
        self.transformer=transformer
    def fit(self,X,y):
        if type(X)!=pd.DataFrame:
            X=pd.DataFrame(X)
        self.X_dtypes_=dict(X.dtypes)
        
        #self.obj_idx=[i for i,(var,dtype) in enumerate(self.X_dtypes_.items()) if dtype=='object']
        
        
        self.obj_idx_=[i for i,(var,dtype) in enumerate(self.X_dtypes_.items()) if dtype=='object']
        self.float_idx_=[i for i in range(X.shape[1]) if i not in self.obj_idx_]
        self.cat_list=[X.iloc[:,idx].unique() for idx in self.obj_idx_]
        return self
        
    def transform(self,X,y=None):
        if type(X)!=pd.DataFrame:
            X=pd.DataFrame(X)
        
        cat_encoder=OneHotEncoder(categories=self.cat_list,sparse=False,) # drop='first'
        xvars=list(X.columns)
        if type(self.strategy) is str:
            if self.strategy=='pass-through':
                numeric_T=('no_transform',none_T(),self.float_idx_)
                categorical_T=('cat_drop_hot',cat_encoder,self.obj_idx_)
            if self.strategy=='drop_row':
                X=X.dropna(axis=0) # overwrite it
                
                numeric_T=('no_transform',none_T(),self.float_idx_)
                categorical_T=('cat_drop_hot',cat_encoder,self.obj_idx_)
                
            if self.strategy=='impute_middle':
                numeric_T=('num_imputer', SimpleImputer(strategy='mean'),self.float_idx_)
                cat_imputer=make_pipeline(SimpleImputer(strategy='most_frequent'),cat_encoder)
                categorical_T=('cat_imputer',cat_imputer,self.obj_idx_)
            if self.strategy[:10]=='impute_knn':
                if len(self.strategy)==10:
                    k=5
                else:
                    k=int(''.join([char for char in self.strategy[10:] if char.isdigit()])) #extract k from the end
                numeric_T=('num_imputer', KNNImputer(strategy='mean'),self.float_idx_)
                cat_imputer=make_pipeline(SimpleImputer(strategy='most_frequent'),cat_encoder)
                categorical_T=('cat_imputer',cat_imputer,self.obj_idx_)
        T=ColumnTransformer(transformers=[numeric_T,categorical_T])
        T.fit(X,y)
        X=T.transform(X)
        #print(X)
        return X