import logging, logging.handlers,os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV, LinearRegression, Lars
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline,FeatureUnion,Pipeline
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


from vb_transformers import none_T#,featureNameExtractor
#,VBHelper,shrinkBigKTransformer,logminus_T,exp_T,logminplus1_T,logp1_T,missingValHandler,dropConst
#from vb_helper import myLogger

class myLogger:
    def __init__(self,name=None):
        if name is None:
            name='missingval.log'
        else:
            if name[-4:]!='.log':
                name+='.log'
        logdir=os.path.join(os.getcwd(),'log'); 
        if not os.path.exists(logdir):os.mkdir(logdir)
        handlername=os.path.join(logdir,name)
        logging.basicConfig(
            handlers=[logging.handlers.RotatingFileHandler(handlername, maxBytes=10**7, backupCount=100)],
            level=logging.DEBUG,
            format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
            datefmt='%Y-%m-%dT%H:%M:%S')
        self.logger = logging.getLogger(handlername)


class missingValHandler(BaseEstimator,TransformerMixin,myLogger):
    # https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html#use-columntransformer-by-selecting-column-by-names
    def __init__(self,prep_dict=None):
    #def __init__(self,strategy='drop_row',transformer=None,cat_idx=None,):
        #self.strategy=strategy
        #self.transformer=transformer
        #self.cat_idx=cat_idx
        self.prep_dict=prep_dict
        #print(prep_dict)
        self.setPrepDictAttrs()
        myLogger.__init__(self,name='missingValHandler.log')
    
    def setPrepDictAttrs(self):
        if 'cat_approach' in self.prep_dict:
            self.cat_approach=self.prep_dict['cat_approach']
        else:
            self.cat_approach='separate'
            
        if 'impute_strategy' in self.prep_dict:
            self.strategy=self.prep_dict['impute_strategy']
        else:self.strategy='drop_row'
        if 'cat_idx' in self.prep_dict:
            self.cat_idx=self.prep_dict['cat_idx']
        else:self.cat_idx=None
            
        
    def fit(self,X,y=None):
        if type(X)!=pd.DataFrame:
            X=pd.DataFrame(X)
        if self.cat_idx is None:
            self.X_dtypes_=dict(X.dtypes)
            if 'object' in list(self.X_dtypes_.values()):
                self.obj_idx_=[i for i,(var,dtype) in enumerate(self.X_dtypes_.items()) if dtype=='object']
            else:
                self.obj_idx=[]
        else:
            self.obj_idx_=self.cat_idx
        self.float_idx_=[i for i in range(X.shape[1]) if i not in self.obj_idx_]
        
        self.cat_list_=[X.iloc[:,idx].unique() for idx in self.obj_idx_]
        x_nan_count=X.isnull().sum().sum() # sums by column and then across columns
        try:
            y_nan_count=y.isnull().sum().sum()
        except:
            try:y_nan_count=np.isnan(y).sum()
            except:
                if not y is None:
                    y_nan_count='error'
                    self.logger.exception(f'error summing nulls for y, type(y):{type(y)}')
                else:
                    y_nan_count='n/a'
                    pass
        self.logger.info(f'x_nan_count:{x_nan_count}, y_nan_count:{y_nan_count}')
        
        ###########
        cat_encoder=OneHotEncoder(categories=self.cat_list_,sparse=False,) # drop='first'
    
            
        if type(self.strategy) is str:
            if self.strategy=='drop':
                assert False, 'develop drop columns with >X% missing vals then drop rows with missing vals'
                
            if self.strategy=='pass-through':
                numeric_T=('no_transform',none_T(),self.float_idx_)
                categorical_T=('cat_onehot',cat_encoder,self.obj_idx_)
            if self.strategy=='drop_row':
                X=X.dropna(axis=0) # overwrite it
                
                numeric_T=('no_transform',none_T(),self.float_idx_)
                categorical_T=('cat_onehot',cat_encoder,self.obj_idx_)
                
            if self.strategy=='impute_middle':
                numeric_T=('num_imputer', SimpleImputer(strategy='mean'),self.float_idx_)
                cat_imputer=make_pipeline(SimpleImputer(strategy='most_frequent'),cat_encoder)
                categorical_T=('cat_imputer',cat_imputer,self.obj_idx_)
            if self.strategy[:10]=='impute_knn':
                if len(self.strategy)==10:
                    k=5
                else:
                    k=int(''.join([char for char in self.strategy[10:] if char.isdigit()])) #extract k from the end
                numeric_T=('num_imputer', KNNImputer(n_neighbors=k),self.float_idx_)
                cat_imputer=make_pipeline(SimpleImputer(strategy='most_frequent'),cat_encoder)
                categorical_T=('cat_imputer',cat_imputer,self.obj_idx_)
            if self.strategy.lower()=="iterativeimputer":
                numeric_T=('num_imputer', IterativeImputer(),self.float_idx_)
                cat_imputer=make_pipeline(SimpleImputer(strategy='most_frequent'),cat_encoder)
                categorical_T=('cat_imputer',cat_imputer,self.obj_idx_)
        if len(self.obj_idx_)==0:
            self.T_=numeric_t[1]
            self.T_.fit(X,y)
        elif self.cat_approach=='together':
            self.T_=ColumnTransformer(
                transformers=[('no_transform',none_T(),self.float_idx_),('cat_onehot',cat_encoder,self.obj_idx_)]
            )
            self.T_.fit(X,y)
            X=self.T_.transform(X) #makes numeric
            self.T1_=numeric_T[1]
            self.T1_.fit(X,y)
        else:
            self.T_=ColumnTransformer(transformers=[numeric_T,categorical_T])
            self.T_.fit(X,y)
        
        return self
    
    def get_feature_names(self,input_features=None):
        cat_feat=[input_features[i]for i in self.obj_idx_]
        float_feat=[input_features[i]for i in self.float_idx_]
        output_features=float_feat
        cat_T=self.T_.transformers_[1][1]
        if type(cat_T) is Pipeline:
            num_cat_feat=cat_T['onehotencoder'].get_feature_names(cat_feat)
            num_cat_feat__=[]
            for name in num_cat_feat:
                for c_idx_l,char in enumerate(name[::-1]):
                    c_idx=len(name)-c_idx_l
                    if char=='_':
                        num_cat_feat__.append(name[:c_idx]+'_'+name[c_idx:])
                        break
                        
            output_features.extend(num_cat_feat__)
        else:
            output_features.extend(cat_T.get_feature_names(cat_feat))
        return output_features
        #return featureNameExtractor(self.T_,input_features=input_features).run()
    
    
    
    def transform(self,X,y=None):
        
        if self.strategy=='drop_row':
            if type(X)!=pd.DataFrame:
                X=pd.DataFrame(X)
            X=X.dropna(axis=0)  
        X=self.T_.transform(X)
        if len(self.obj_idx_)>0 and self.cat_approach=='together':
            X=self.T1_.transform(X)
        
        x_nan_count=np.isnan(X).sum() # sums by column and then across columns
        """try:
            y_nan_count=y.isnull().sum().sum()
        except:
            y_nan_count='error'
            self.logger.exception('error summing nulls for y')"""
        if x_nan_count>0:
            self.logger.info(f'x_nan_count is non-zero! x_nan_count:{x_nan_count}')
        #print(X)
        return X
    
    
    
