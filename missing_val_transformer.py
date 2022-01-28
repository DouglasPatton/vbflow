import logging, logging.handlers,os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV, LinearRegression, Lars
from sklearn.ensemble import GradientBoostingRegressor
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


    #unpacking the dictionary prep_dict
    def setPrepDictAttrs(self):
        if 'cat_approach' in self.prep_dict: #whether to impute categorical variables separately from floats
            self.cat_approach=self.prep_dict['cat_approach']
        else:
            self.cat_approach='separate'
            
        if 'impute_strategy' in self.prep_dict: #imputation strategy, drop_row is the default
            self.strategy=self.prep_dict['impute_strategy']
        else:self.strategy='drop_row'

        if 'cat_idx' in self.prep_dict: #getting the column numbers of the categorical variables
            self.cat_idx=self.prep_dict['cat_idx']
        else:self.cat_idx=None

    # learns how to binarize and impute based off the data its fed
    def fit(self,X,y=None):
        if type(X)!=pd.DataFrame: #if X is not a dataframe, make it so
            X=pd.DataFrame(X)
        if self.cat_idx is None: #if no categorical variable locations supplied, figure out their locations
            self.X_dtypes_=dict(X.dtypes)
            if 'object' in list(self.X_dtypes_.values()):
                self.obj_idx_=[i for i,(var,dtype) in enumerate(self.X_dtypes_.items()) if dtype=='object']
            else:
                self.obj_idx_=[]
        else:
            self.obj_idx_=self.cat_idx

        self.float_idx_=[i for i in range(X.shape[1]) if i not in self.obj_idx_] #any non-categorical variable is a float
        self.cat_list_=[X.iloc[:,idx].unique() for idx in self.obj_idx_] #figuring out the levels of each categorical variable
        x_nan_count=X.isnull().sum().sum() #sums all missing values in X
        #the following block of code records the missing values in y if y is supplied
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

        cat_encoder=OneHotEncoder(categories=self.cat_list_,sparse=False,) # scikit learn function that does binarization
        # of the categorical variables; drop='first' could be used at the end of OneHotEncoder() to drop the first specified level of the category

        if type(self.strategy) is str: #this block is initializing an imputation strategy
            if self.strategy=='drop': #code for this strategy not written yet
                assert False, 'develop drop columns with >X% missing vals then drop rows with missing vals'
                
            if self.strategy=='pass-through': #option for estimators that can handle missing values
                numeric_T=('no_transform',none_T(),self.float_idx_) #tuple has the label, the transformer, and what columns to apply transformer to
                categorical_T=('cat_onehot',cat_encoder,self.obj_idx_) #NEEDS WORK; maybe cat_encoder could be changed to none_T()

            if self.strategy=='drop_row': #dropping rows with missing values in any variable
                assert False, "NEEDS WORK"
                X=X.dropna(axis=0) #overwrite X
                numeric_T=('no_transform',none_T(),self.float_idx_)
                categorical_T=('cat_onehot',cat_encoder,self.obj_idx_)
                
            if self.strategy=='impute_middle': #uses mean (numerical) or most frequent (categorical) for imputation
                numeric_T=('num_imputer', SimpleImputer(strategy='mean'),self.float_idx_)
                cat_imputer=make_pipeline(SimpleImputer(strategy='most_frequent'),cat_encoder) #chains together imputation and binarization
                categorical_T=('cat_imputer',cat_imputer,self.obj_idx_)

            if self.strategy[:10]=='impute_knn': #uses k nearest neighbor (numerical) or most frequent (categorical)
                if len(self.strategy)==10:
                    k=5
                else:
                    k=int(''.join([char for char in self.strategy[10:] if char.isdigit()])) #extract k from the end of self.strategy string
                numeric_T=('num_imputer', KNNImputer(n_neighbors=k),self.float_idx_)
                cat_imputer=make_pipeline(SimpleImputer(strategy='most_frequent'),cat_encoder)
                categorical_T=('cat_imputer',cat_imputer,self.obj_idx_)

            if self.strategy.lower()=="iterativeimputer": #uses an iterative imputer - MICE
                #default is Bayesian Ridge Regressor, but that stores way too much info; seems like overkill; switched to linear regression
                #values for max_iter and tol were specified here to ascertain why disk space storage was so large
                numeric_T=('num_imputer', IterativeImputer(estimator=LinearRegression(),max_iter=10,tol=.01),self.float_idx_)
                cat_imputer=make_pipeline(SimpleImputer(strategy='most_frequent'),cat_encoder)
                categorical_T=('cat_imputer',cat_imputer,self.obj_idx_)

        if len(self.obj_idx_)==0: #if no categorical variables, do just numeric transforms
            self.T_=ColumnTransformer(transformers=[numeric_T])
            self.T_.fit(X,y)
        elif len(self.float_idx_)==0: #if no numeric variables, do just categorical transforms
            self.T_=ColumnTransformer(transformers=[categorical_T])
            self.T_.fit(X,y)
        elif self.cat_approach=='together': #categorical variables are binarized and then all variables are imputed as numeric
            self.T_=ColumnTransformer(
                transformers=[('no_transform',none_T(),self.float_idx_),('cat_onehot',cat_encoder,self.obj_idx_)]
            )
            self.T_.fit(X,y)
            X=self.T_.transform(X) #binarizes categorical variables and establishes variable names
            self.T1_=numeric_T[1] #takes the middle value (transformer) of the listed tuple
            self.T1_.fit(X,y) #performs the imputation
        else: #if there are both categorical and numeric variables, and the cat_approach is not 'together',
            #imputation happens separately for categorical and numeric variables
            self.T_=ColumnTransformer(transformers=[numeric_T,categorical_T])
            self.T_.fit(X,y)
        
        return self

    def get_feature_names(self,input_features=None): #unpacks categorical variable names
        cat_feat=[input_features[i]for i in self.obj_idx_] #getting categorical variable names
        float_feat=[input_features[i]for i in self.float_idx_] #getting float variable names
        output_features=float_feat #start a new list of variables, always float/numeric first
        if len(self.obj_idx_)>0: #if there are categorical variables
            if len(self.float_idx_)>0: #if there are float variables
                cat_T=self.T_.transformers_[1][1] #get the second transformer tuple, then get the second item
            else:
                cat_T=self.T_.transformers_[0][1]#get the first transformer tuple (b/c no cat vars), then get the second item
            if type(cat_T) is Pipeline: #categorical transformers that have two steps (binarize then impute) are defined as a Pipeline
                num_cat_feat=cat_T['onehotencoder'].get_feature_names(cat_feat)
                num_cat_feat__=[]
                #this loop is finding the last underscore in each cat var name and adds another underscore
                #TEST THIS to make sure it works with all sorts of imported categorical variable names and level names
                for name in num_cat_feat: #concatenating the name of a cat variable to the various levels of that cat variable, for every cat variable
                    for c_idx_l,char in enumerate(name[::-1]): #-1 makes the character iteration occur from right to left
                        c_idx=len(name)-c_idx_l
                        if char=='_':
                            num_cat_feat__.append(name[:c_idx]+'_'+name[c_idx:])
                            break

                output_features.extend(num_cat_feat__)
            else: #TEST THIS to see if it works with non-pipeline categorical transformer
                output_features.extend(cat_T.get_feature_names(cat_feat))
        return output_features
        #return featureNameExtractor(self.T_,input_features=input_features).run()

    def transform(self,X,y=None): #binarize and impute the data based on fit's findings
        
        if self.strategy=='drop_row': #NEEDS ATTENTION, may not work
            if type(X)!=pd.DataFrame: #makes X a data frame if its not
                X=pd.DataFrame(X)
            X=X.dropna(axis=0)  
        X=self.T_.transform(X) #binarize/impute X
        if len(self.obj_idx_)>0 and len(self.float_idx_)>0 and self.cat_approach=='together':
            X=self.T1_.transform(X) #this is imputing all variables, which have been made floats
        
        x_nan_count=np.isnan(X).sum() #sums missing values by column and then across columns
        #commented code for debugging
        '''try:
            y_nan_count=y.isnull().sum().sum()
        except:
            y_nan_count='error'
            self.logger.exception('error summing nulls for y')'''
        if x_nan_count>0:
            self.logger.info(f'x_nan_count is non-zero! x_nan_count:{x_nan_count}')
        #print(X)
        return X