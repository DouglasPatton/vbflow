import numpy as np
from scipy.optimize import least_squares
from sklearn.base import BaseEstimator, TransformerMixin,RegressorMixin
from sklearn.datasets import make_regression
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.experimental import enable_hist_gradient_boosting  
from sklearn.ensemble import GradientBoostingRegressor,HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Lars,Lasso,LassoCV,LassoLarsCV,ElasticNetCV
from sklearn.svm import LinearSVR, SVR
from sklearn.preprocessing import StandardScaler, FunctionTransformer, PolynomialFeatures, OneHotEncoder, PowerTransformer
from sklearn.model_selection import cross_validate, train_test_split, RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import TransformedTargetRegressor
import matplotlib.pyplot as plt
from vb_helper import myLogger,VBHelper
from vb_transformers import shrinkBigKTransformer,logminus_T,exp_T,logminplus1_T,none_T,log_T, logp1_T,dropConst,columnBestTransformer
from missing_val_transformer import missingValHandler
from nonlinear_stacker import stackNonLinearTransforms
import os
import pandas as pd
from vb_cross_validator import regressor_q_stratified_cv

try:
    import daal4py.sklearn
    daal4py.sklearn.patch_sklearn()
except:
    print('no daal4py')
 
class BaseHelper:
    def __init__(self):
        pass
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
  

'''
class RegularizedFlexibleEstimator(BaseEstimator,RegressorMixin,myLogger):
    def __init__(self,flex_kwargs={'form':'exp(XB)','regularize':'l1'}):
        self.flex_kwargs=flex_kwargs
        
    def expXB(self,B,X):
        Bconst=B[0]
        Betas=B[1:]
        return np.exp(Bconst+(X*Betas).sum(axis=1))
    
    def est_residuals(self,B,X,y):
        if regularize in self.flex_kwargs:
            res=self.est_(B,X)-y
            sgn=np.ones_like(res)
            sgn[res<0]=-1
            #if self.flex_kwargs['regularize']=='l1':
            #    res+=B.sum()*
            return 
        else:
            return self.est_(B,X)-y
    
    def fit(self,X,y):
        if self.flex_kwargs['form']=='exp(XB)':
            self.est_=self.expXB
        #https://scipy-cookbook.readthedocs.io/items/robust_regression.html
        self.fit_est_=least_squares(self.est_residuals, np.ones(X.shape[1]),args=(X, y))# loss='soft_l1', f_scale=0.1, )
        return self
    
    """def score(self,X,y):
        #negative mse
        return mean_squared_error(self.predict(X),y)"""
    
    def predict(self,X):
        return self.est_(self.fit_est_.x,X)
'''



class FlexibleEstimator(BaseEstimator,RegressorMixin,myLogger):
    def __init__(self,form='expXB',robust=False,shift=True,scale=True):
        self.form=form
        self.robust=robust
        self.shift=shift
        self.scale=scale
    
    def linear(self,B,X):
        Bconst=B[0]
        Betas=B[1:]
        y=Bconst+(X@Betas)
        return np.nan_to_num(y,nan=1e298)
    
    def powXB(self,B,X):
        param_idx=0
        if self.shift:
            Bshift=B[param_idx]
            param_idx+=1
        else:
            Bshift=0
        if self.scale:
            Bscale=B[param_idx]
            param_idx+=1
        else:
            Bscale=1
        Bexponent=B[param_idx]
        param_idx+=1
        Bconst=B[param_idx]
        param_idx+=1
        Betas=B[param_idx:]
        
        y=Bshift+Bscale*(Bconst+(X@Betas))**(int(Bexponent)) 
        return np.nan_to_num(y,nan=1e290)
    
    def expXB(self,B,X):
        param_idx=0
        if self.shift:
            Bshift=B[param_idx]
            param_idx+=1
        else:
            Bshift=0
        if self.scale:
            Bscale=B[param_idx]
            param_idx+=1
        else:
            Bscale=1
        Bconst=B[param_idx]
        param_idx+=1
        Betas=B[param_idx:]
        y=Bshift+Bscale*np.exp(Bconst+(X@Betas))
        return np.nan_to_num(y,nan=1e298)
    
    def est_residuals(self,B,X,y):
        return self.est_(B,X)-y
    
    def fit(self,X,y):
        if self.form=='expXB':
            self.est_=self.expXB
            self.k=X.shape[1]+1 # constant
        elif self.form=='powXB':
            self.est_=self.powXB
            self.k=X.shape[1]+2 # constant & exponent
        elif self.form=='linear':
            self.est=self.linear
            self.k=X.shape[1]+1 # constant
        if not self.form=='linear':
            if self.scale:
                self.k+=1
            if self.shift:
                self.k+=1
        #https://scipy-cookbook.readthedocs.io/items/robust_regression.html
        if self.robust:
            self.fit_est_=least_squares(self.est_residuals, np.ones(self.k),args=(X, y),loss='soft_l1', f_scale=10,)# 
        else:
            self.fit_est_=least_squares(self.est_residuals, np.ones(self.k),args=(X, y))# 
        return self
    
    """def score(self,X,y):
        #negative mse
        return mean_squared_error(self.predict(X),y)"""
    
    def predict(self,X):
        B=self.fit_est_.x
        return self.est_(B,X)
        
    
            
        

class FlexiblePipe(BaseEstimator,TransformerMixin,myLogger,BaseHelper):
    def __init__(
        self,functional_form_search =False,
        impute_strategy='impute_knn5',gridpoints=4,
        cv_strategy='quantile',groupcount=5,bestT=False,
        cat_idx=None,float_idx=None,flex_kwargs={}
        ):
        
        myLogger.__init__(self,name='l1lars.log')
        self.logger.info('starting l1lars logger')
        self.functional_form_search=functional_form_search
        self.gridpoints=gridpoints
        self.cv_strategy=cv_strategy
        self.groupcount=groupcount
        self.bestT=bestT
        self.cat_idx=cat_idx
        self.float_idx=float_idx
        self.impute_strategy=impute_strategy
        self.flex_kwargs=flex_kwargs
        BaseHelper.__init__(self)
        
    def get_estimator(self,):
        if self.cv_strategy:
            inner_cv=regressor_q_stratified_cv(n_splits=10,n_repeats=5, strategy=self.cv_strategy,random_state=0,groupcount=self.groupcount)
        
        else:
            inner_cv=RepeatedKFold(n_splits=10, n_repeats=5, random_state=0)
        
        steps=[
            ('prep',missingValHandler(strategy=self.impute_strategy,cat_idx=self.cat_idx)),
            ('scaler',StandardScaler()),
            ('select',shrinkBigKTransformer(max_k=4)),
            ('reg',FlexibleEstimator(**self.flex_kwargs))
        ]
        if self.bestT:
            steps=[steps[0],('select',columnBestTransformer(float_k=len(self.float_idx))),*steps[1:]]
        pipe=Pipeline(steps=steps)
        param_grid={'select__k_share':np.linspace(0.2,1,self.gridpoints*2)}
        if self.functional_form_search:
            param_grid['reg__form']=['powXB','expXB','linear']
            

        outerpipe=GridSearchCV(pipe,param_grid=param_grid)
        return outerpipe
    
class L1Lars(BaseEstimator,TransformerMixin,myLogger,BaseHelper):
    def __init__(self,impute_strategy='impute_knn5',gridpoints=4,cv_strategy='quantile',groupcount=5,bestT=False,cat_idx=None,float_idx=None):
        myLogger.__init__(self,name='l1lars.log')
        self.logger.info('starting l1lars logger')
        self.gridpoints=gridpoints
        self.cv_strategy=cv_strategy
        self.groupcount=groupcount
        self.bestT=bestT
        self.cat_idx=cat_idx
        self.float_idx=float_idx
        self.impute_strategy=impute_strategy
        BaseHelper.__init__(self)
        
    """ def fit(self,X,y):
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
        return self.est_.predict(X)"""
    
    def get_estimator(self,):
        if self.cv_strategy:
            inner_cv=regressor_q_stratified_cv(n_splits=10,n_repeats=5, strategy=self.cv_strategy,random_state=0,groupcount=self.groupcount)
        
        else:
            inner_cv=RepeatedKFold(n_splits=10, n_repeats=5, random_state=0)
        #gridpoints=self.gridpoints
        #param_grid={'l1_ratio':np.logspace(-2,-.03,gridpoints*2)}
        steps=[
            ('prep',missingValHandler(strategy=self.impute_strategy,cat_idx=self.cat_idx)),
            ('reg',LassoLarsCV(cv=inner_cv,))]
        if self.bestT:
            steps=[steps[0],('select',columnBestTransformer(float_k=len(self.float_idx))),*steps[1:]]
        pipe=Pipeline(steps=steps)
        
        return pipe
    
class GBR(BaseEstimator,TransformerMixin,myLogger,BaseHelper):
    def __init__(self,impute_strategy='impute_knn5',bestT=False,cat_idx=None,float_idx=None):
        myLogger.__init__(self,name='gbr.log')
        self.logger.info('starting gradient_boosting_reg logger')
        self.bestT=bestT
        self.cat_idx=cat_idx
        self.float_idx=float_idx
        self.impute_strategy=impute_strategy
        BaseHelper.__init__(self)
    def get_estimator(self):
        steps=[
            ('prep',missingValHandler(strategy=self.impute_strategy,cat_idx=self.cat_idx)),
            ('reg',GradientBoostingRegressor())
        ]
        if self.bestT:
            steps=[steps[0],('select',columnBestTransformer(float_k=len(self.float_idx))),*steps[1:]]
        return Pipeline(steps=steps)
        
class HGBR(BaseEstimator,TransformerMixin,myLogger,BaseHelper):
    def __init__(self,cat_idx=None,float_idx=None):
        myLogger.__init__(self,name='HGBR.log')
        self.logger.info('starting histogram_gradient_boosting_reg logger')
        self.cat_idx=cat_idx
        self.float_idx=float_idx
        BaseHelper.__init__(self)
    def get_estimator(self):
        steps=[
            ('prep',missingValHandler(strategy='pass-through',cat_idx=self.cat_idx)),
            ('reg',HistGradientBoostingRegressor())
        ]
        return Pipeline(steps=steps)


class ENet(BaseEstimator,TransformerMixin,myLogger,BaseHelper):
    def __init__(self,impute_strategy='impute_knn5',gridpoints=4,cv_strategy='quantile',groupcount=5,float_idx=None,cat_idx=None,bestT=False):
        myLogger.__init__(self,name='enet.log')
        self.logger.info('starting enet logger')
        self.gridpoints=gridpoints
        self.cv_strategy=cv_strategy
        self.groupcount=groupcount
        self.float_idx=float_idx
        self.cat_idx=cat_idx
        self.bestT=bestT
        self.impute_strategy=impute_strategy
        BaseHelper.__init__(self)

    def get_estimator(self,):
        if self.cv_strategy:
            inner_cv=regressor_q_stratified_cv(n_splits=10,n_repeats=5, strategy=self.cv_strategy,random_state=0,groupcount=self.groupcount)
        
        else:
            inner_cv=RepeatedKFold(n_splits=10, n_repeats=5, random_state=0)
        gridpoints=self.gridpoints
        param_grid={'l1_ratio':np.logspace(-2,-.03,gridpoints)}
        steps=[
            ('prep',missingValHandler(strategy=self.impute_strategy,cat_idx=self.cat_idx)),
            #('shrink_k1',shrinkBigKTransformer(selector=LassoLarsCV(cv=inner_cv,max_iter=32))), # retain a subset of the best original variables
            #('polyfeat',PolynomialFeatures(interaction_only=0,degree=2d)), # create interactions among them
            
            #('drop_constant',dropConst()),
            #('shrink_k2',shrinkBigKTransformer(selector=LassoLarsCV(cv=inner_cv,max_iter=64))),
            ('scaler',StandardScaler()),
            ('reg',GridSearchCV(ElasticNetCV(cv=inner_cv,normalize=False),param_grid=param_grid))]
            #('reg',ElasticNetCV(cv=inner_cv,normalize=True))]
            
        if self.bestT:
            steps=[steps[0],('select',columnBestTransformer(float_k=len(self.float_idx))),*steps[1:]]
        pipe=Pipeline(steps=steps)
        
        return pipe

class RBFSVR(BaseEstimator,TransformerMixin,myLogger,BaseHelper):
    def __init__(self,impute_strategy='impute_knn5',gridpoints=4,cv_strategy='quantile',groupcount=5,float_idx=None,cat_idx=None,bestT=False):
        myLogger.__init__(self,name='LinRegSupreme.log')
        self.logger.info('starting LinRegSupreme logger')
        self.gridpoints=gridpoints
        self.cv_strategy=cv_strategy
        self.groupcount=groupcount
        self.bestT=bestT
        self.cat_idx=cat_idx
        self.float_idx=float_idx
        self.impute_strategy=impute_strategy
        BaseHelper.__init__(self)
        
    """def fit(self,X,y):
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
        return self.est_.predict(X)"""
    
    def get_estimator(self,):
        if self.cv_strategy:
            inner_cv=regressor_q_stratified_cv(n_splits=10,n_repeats=5, strategy=self.cv_strategy,random_state=0,groupcount=self.groupcount)
        
        else:
            inner_cv=RepeatedKFold(n_splits=10, n_repeats=5, random_state=0)
        gridpoints=self.gridpoints
        param_grid={'C':np.logspace(-2,2,gridpoints),
                   'gamma':np.logspace(-2,0.5,gridpoints)}
        steps=[
            ('prep',missingValHandler(strategy=self.impute_strategy,cat_idx=self.cat_idx)),
            #('shrink_k1',shrinkBigKTransformer(selector=LassoLarsCV(cv=inner_cv,max_iter=32))), # retain a subset of the best original variables
            #('polyfeat',PolynomialFeatures(interaction_only=0,degree=2)), # create interactions among them
            
            #('drop_constant',dropConst()),
            #('shrink_k2',shrinkBigKTransformer(selector=LassoLarsCV(cv=inner_cv,max_iter=64))),
            ('scaler',StandardScaler()),
            ('reg',GridSearchCV(SVR(kernel='rbf',cache_size=10000,tol=1e-4,max_iter=5000),param_grid=param_grid))]
        if self.bestT:
            steps=[steps[0],('select',columnBestTransformer(float_k=len(self.float_idx))),*steps[1:]]
        pipe=Pipeline(steps=steps)
        
        outerpipe=pipe
        return outerpipe



class LinSVR(BaseEstimator,TransformerMixin,myLogger,BaseHelper):
    def __init__(self,impute_strategy='impute_knn5',gridpoints=4,cv_strategy='quantile',groupcount=5,bestT=False,cat_idx=None,float_idx=None):
        myLogger.__init__(self,name='LinRegSupreme.log')
        self.logger.info('starting LinRegSupreme logger')
        self.gridpoints=gridpoints
        self.cv_strategy=cv_strategy
        self.groupcount=groupcount
        self.bestT=bestT
        self.cat_idx=cat_idx
        self.float_idx=float_idx
        self.impute_strategy=impute_strategy
        BaseHelper.__init__(self)
        
    """def fit(self,X,y):
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
        return self.est_.predict(X)"""
    
    def get_estimator(self,):
        if self.cv_strategy:
            inner_cv=regressor_q_stratified_cv(n_splits=10,n_repeats=5, strategy=self.cv_strategy,random_state=0,groupcount=self.groupcount)
        
        else:
            inner_cv=RepeatedKFold(n_splits=10, n_repeats=5, random_state=0)
        gridpoints=self.gridpoints
        param_grid={'C':np.logspace(-2,4,gridpoints)}
        steps=[
            ('prep',missingValHandler(strategy=self.impute_strategy,cat_idx=self.cat_idx)),
            #('shrink_k1',shrinkBigKTransformer(selector=LassoLarsCV(cv=inner_cv,max_iter=32))), # retain a subset of the best original variables
            ('polyfeat',PolynomialFeatures(interaction_only=0,degree=2)), # create interactions among them
            
            ('drop_constant',dropConst()),
            ('shrink_k2',shrinkBigKTransformer(selector=LassoLarsCV(cv=inner_cv,max_iter=64))),
            ('scaler',StandardScaler()),
            ('reg',GridSearchCV(LinearSVR(random_state=0,tol=1e-4,max_iter=1000),param_grid=param_grid))]
        if self.bestT:
            steps=[steps[0],('select',columnBestTransformer(float_k=len(self.float_idx))),*steps[1:]]
        pipe=Pipeline(steps=steps)
        
        outerpipe=pipe
        return outerpipe

        
class LinRegSupreme(BaseEstimator,TransformerMixin,myLogger,BaseHelper):
    def __init__(self,impute_strategy='impute_knn5',gridpoints=4,cv_strategy='quantile',groupcount=5,bestT=False,cat_idx=None,float_idx=None):
        myLogger.__init__(self,name='LinRegSupreme.log')
        self.logger.info('starting LinRegSupreme logger')
        self.gridpoints=gridpoints
        self.cv_strategy=cv_strategy
        self.groupcount=groupcount
        self.bestT=bestT
        self.cat_idx=cat_idx
        self.float_idx=float_idx
        self.impute_strategy=impute_strategy
        BaseHelper.__init__(self)
    
    
    def get_estimator(self,):
        if self.cv_strategy:
            inner_cv=regressor_q_stratified_cv(n_splits=10,n_repeats=5, strategy=self.cv_strategy,random_state=0,groupcount=self.groupcount)
        
        else:
            inner_cv=RepeatedKFold(n_splits=10, n_repeats=5, random_state=0)
        gridpoints=self.gridpoints
        transformer_list=[none_T(),log_T(),logp1_T()]#,logp1_T()] # log_T()]#
        steps=[
            ('prep',missingValHandler(strategy=self.impute_strategy)),
            #('nonlin_stacker',stackNonLinearTransforms()),
            #,
            #('shrink_k1',shrinkBigKTransformer(selector=Lasso())), # retain a subset of the best original variables
            ('shrink_k1',shrinkBigKTransformer(selector=LassoLarsCV(cv=inner_cv,max_iter=32))), # retain a subset of the best original variables
            ('polyfeat',PolynomialFeatures(interaction_only=0,degree=2)), # create interactions among them
            
            ('drop_constant',dropConst()),
            ('shrink_k2',shrinkBigKTransformer(selector=LassoLarsCV(cv=inner_cv,max_iter=64))), # pick from all of those options
            ('reg',LinearRegression())]
        if self.bestT:
            steps=[steps[0],('select',columnBestTransformer(float_k=len(self.float_idx))),*steps[1:]]

        X_T_pipe=Pipeline(steps=steps)
        #inner_cv=regressor_stratified_cv(n_splits=10,n_repeats=2,shuffle=True)
        


        Y_T_X_T_pipe=Pipeline(steps=[('ttr',TransformedTargetRegressor(regressor=X_T_pipe))])
        Y_T__param_grid={
            'ttr__transformer':transformer_list,
            'ttr__regressor__polyfeat__degree':[2],
            #'ttr__regressor__shrink_k1__selector__alpha':np.logspace(-1.5,1.5,gridpoints),
            #'ttr__regressor__shrink_k2__selector__alpha':list(np.logspace(-2,1.3,gridpoints*2)),
            #'ttr__regressor__shrink_k2__selector__l1_ratio':list(np.linspace(0.05,.95,gridpoints)),
            #'ttr__regressor__shrink_k1__max_k':[self.k_//gp for gp in range(1,gridpoints+1,2)],
            #'ttr__regressor__shrink_k1__k_share':list(np.linspace(1/self.k_,1,gridpoints)),
            #'ttr__regressor__prep__strategy':['impute_middle','impute_knn_10']
            'ttr__regressor__prep__strategy':['impute_middle','impute_knn_10']
        }
        #lin_reg_Xy_transform=GridSearchCV(Y_T_X_T_pipe,param_grid=Y_T__param_grid,cv=inner_cv,n_jobs=10,refit='neg_mean_squared_error')
        #lin_reg_Xy_transform=GridSearchCV(Y_T_X_T_pipe,param_grid=Y_T__param_grid,cv=inner_cv,n_jobs=4,refit='neg_mean_squared_error')
        lin_reg_Xy_transform=X_T_pipe
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