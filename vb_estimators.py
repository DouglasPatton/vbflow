import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import make_regression
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.linear_model import ElasticNet, LinearRegression, Lars, TweedieRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.preprocessing import StandardScaler, FunctionTransformer, PolynomialFeatures, OneHotEncoder, PowerTransformer
from sklearn.model_selection import cross_validate, train_test_split, RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor
import matplotlib.pyplot as plt
from vb_helper import myLogger,VBHelper,shrinkBigKTransformer,logminus_T,exp_T,logminplus1_T,none_T, logp1_T,dropConst
from missing_val_transformer import missingValHandler
import os
import pandas as pd

try:
    import daal4py.sklearn
    daal4py.sklearn.patch_sklearn()
except:
    print('no daal4py')
        
        
class LinRegSupreme(BaseEstimator,TransformerMixin,myLogger):
    def __init__(self,gridpoints=3):
        myLogger.__init__(self,name='LinRegSupreme.log')
        self.logger.info('starting LinRegSupreme logger')
        self.gridpoints=gridpoints
        
    def fit(self,X,y):
        self.n_,self.k_=X.shape
        #self.logger(f'self.k_:{self.k_}')
        self.est_=self.get_estimator()
        self.est_.fit(X,y)
        return self
    def transform(self,X,y=None):
        return self.est_.transform(X,y)
    def score(self,X,y):
        return self.est_.score(X,y)
    def predict(self,X):
        return self.est_.predict(X)
    
    def get_estimator(self,):
        gridpoints=self.gridpoints
        transformer_list=[none_T(),logp1_T()]
        steps=[
            ('prep',missingValHandler()),
            ('scaler',StandardScaler()),
            ('shrink_k1',shrinkBigKTransformer()), # retain a subset of the best original variables
            ('polyfeat',PolynomialFeatures(interaction_only=0)), # create interactions among them

            ('drop_constant',dropConst()),
            ('shrink_k2',shrinkBigKTransformer(selector=ElasticNet())), # pick from all of those options
            ('reg',LinearRegression(fit_intercept=1))]


        X_T_pipe=Pipeline(steps=steps)
        inner_cv=RepeatedKFold(n_splits=5, n_repeats=2, random_state=0)


        Y_T_X_T_pipe=Pipeline(steps=[('ttr',TransformedTargetRegressor(regressor=X_T_pipe))])
        Y_T__param_grid={
            'ttr__transformer':transformer_list,
            'ttr__regressor__polyfeat__degree':[2],
            'ttr__regressor__shrink_k2__selector__alpha':np.logspace(-2,2,gridpoints),
            'ttr__regressor__shrink_k2__selector__l1_ratio':np.linspace(0,1,gridpoints),
            'ttr__regressor__shrink_k1__max_k':[self.k_//gp for gp in range(1,gridpoints+1)],
            'ttr__regressor__prep__strategy':['impute_middle','impute_knn_10']
        }
        lin_reg_Xy_transform=GridSearchCV(Y_T_X_T_pipe,param_grid=Y_T__param_grid,cv=inner_cv,n_jobs=-1)

        return lin_reg_Xy_transform
    
    
    
if __name__=="__main__":
    X, y= make_regression(n_samples=30,n_features=5,noise=1)
    lrs=LinRegSupreme()
    lrs.fit(X,y)
    s=lrs.score(X,y)
    print(f'r2 score: {s}')
    cv=cross_validate(lrs,X,y,scoring='r2',cv=2)
    print(cv)
    print(cv['test_score'])