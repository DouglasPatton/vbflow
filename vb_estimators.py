import numpy as np
from scipy.optimize import least_squares
from sklearn.base import BaseEstimator, TransformerMixin,RegressorMixin
from sklearn.datasets import make_regression
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.experimental import enable_hist_gradient_boosting  
from sklearn.ensemble import GradientBoostingRegressor,HistGradientBoostingRegressor,StackingRegressor
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
        self.pipe_=self.get_pipe()
        self.pipe_.fit(X,y)
        return self
    def transform(self,X,y=None):
        return self.pipe_.transform(X,y)
    def score(self,X,y):
        return self.pipe_.score(X,y)
    def predict(self,X):
        return self.pipe_.predict(X)

    

    
    

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
            res=self.pipe_(B,X)-y
            sgn=np.ones_like(res)
            sgn[res<0]=-1
            #if self.flex_kwargs['regularize']=='l1':
            #    res+=B.sum()*
            return 
        else:
            return self.pipe_(B,X)-y
    
    def fit(self,X,y):
        if self.flex_kwargs['form']=='exp(XB)':
            self.pipe_=self.expXB
        #https://scipy-cookbook.readthedocs.io/items/robust_regression.html
        self.fit_est_=least_squares(self.pipe_residuals, np.ones(X.shape[1]),args=(X, y))# loss='soft_l1', f_scale=0.1, )
        return self
    
    """def score(self,X,y):
        #negative mse
        return mean_squared_error(self.predict(X),y)"""
    
    def predict(self,X):
        return self.pipe_(self.fit_est_.x,X)
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
    
    def pipe_residuals(self,B,X,y):
        return self.pipe_(B,X)-y
    
    def fit(self,X,y):
        if self.form=='expXB':
            self.pipe_=self.expXB
            self.k=X.shape[1]+1 # constant
        elif self.form=='powXB':
            self.pipe_=self.powXB
            self.k=X.shape[1]+2 # constant & exponent
        elif self.form=='linear':
            self.pipe_=self.linear
            self.k=X.shape[1]+1 # constant
        if not self.form=='linear':
            if self.scale:
                self.k+=1
            if self.shift:
                self.k+=1
        #https://scipy-cookbook.readthedocs.io/items/robust_regression.html
        if self.robust:
            self.fit_est_=least_squares(self.pipe_residuals, np.ones(self.k),args=(X, y),loss='soft_l1', f_scale=0.1,)# 
        else:
            self.fit_est_=least_squares(self.pipe_residuals, np.ones(self.k),args=(X, y))# 
        return self
    
    """def score(self,X,y):
        #negative mse
        return mean_squared_error(self.predict(X),y)"""
    
    def predict(self,X):
        B=self.fit_est_.x
        return self.pipe_(B,X)
        
    
            
        

class FlexiblePipe(BaseEstimator,RegressorMixin,myLogger,BaseHelper):
    def __init__(
        self,do_prep=True,functional_form_search =False,
        impute_strategy='impute_knn5',gridpoints=4,
        cv_strategy=None,groupcount=None,bestT=False,
        cat_idx=None,float_idx=None,flex_kwargs={}
        ):
        
        myLogger.__init__(self,name='l1lars.log')
        self.logger.info('starting l1lars logger')
        self.do_prep=do_prep
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
        
    def get_pipe(self,):
        if self.cv_strategy:
            inner_cv=regressor_q_stratified_cv(n_splits=10,n_repeats=3, strategy=self.cv_strategy,random_state=0,groupcount=self.groupcount)
        
        else:
            inner_cv=RepeatedKFold(n_splits=10, n_repeats=3, random_state=0)
            
        steps=[
            ('scaler',StandardScaler()),
            ('select',shrinkBigKTransformer(max_k=4)),
            ('reg',FlexibleEstimator(**self.flex_kwargs))
        ]
        if self.bestT:
            steps.insert(0,'xtransform',columnBestTransformer(float_k=len(self.float_idx)))
        
                       
        pipe=Pipeline(steps=steps)
        param_grid={'select__k_share':np.linspace(0.2,1,self.gridpoints*2)}
        if self.functional_form_search:
            param_grid['reg__form']=['powXB','expXB']#,'linear']
            

        outerpipe=GridSearchCV(pipe,param_grid=param_grid)
        if self.do_prep:
            steps=[('prep',missingValHandler(strategy=self.impute_strategy,cat_idx=self.cat_idx)),
                   ('post',outerpipe)]
            outerpipe=Pipeline(steps=steps)
            
        return outerpipe
    
class L1Lars(BaseEstimator,RegressorMixin,myLogger,BaseHelper):
    def __init__(self,do_prep=True,impute_strategy='impute_knn5',
                 gridpoints=4,cv_strategy=None,groupcount=None,
                 bestT=False,cat_idx=None,float_idx=None):
        myLogger.__init__(self,name='l1lars.log')
        self.logger.info('starting l1lars logger')
        self.do_prep=do_prep
        self.gridpoints=gridpoints
        self.cv_strategy=cv_strategy
        self.groupcount=groupcount
        self.bestT=bestT
        self.cat_idx=cat_idx
        self.float_idx=float_idx
        self.impute_strategy=impute_strategy
        BaseHelper.__init__(self)
    
    def get_pipe(self,):
        if self.cv_strategy:
            inner_cv=regressor_q_stratified_cv(n_splits=10,n_repeats=3, strategy=self.cv_strategy,random_state=0,groupcount=self.groupcount)
        
        else:
            inner_cv=RepeatedKFold(n_splits=10, n_repeats=3, random_state=0)
        
        steps=[('reg',LassoLarsCV(cv=inner_cv,))]
        if self.bestT:
            steps.insert(0,'xtransform',columnBestTransformer(float_k=len(self.float_idx)))
        outerpipe=Pipeline(steps=steps)
        
        if self.do_prep:
            steps=[('prep',missingValHandler(strategy=self.impute_strategy,cat_idx=self.cat_idx)),
                   ('post',outerpipe)]
            outerpipe=Pipeline(steps=steps)
        
        return outerpipe
    
class GBR(BaseEstimator,RegressorMixin,myLogger,BaseHelper):
    def __init__(self,do_prep=True,impute_strategy='impute_knn5',cv_strategy=None,bestT=False,cat_idx=None,float_idx=None):
        myLogger.__init__(self,name='gbr.log')
        self.logger.info('starting gradient_boosting_reg logger')
        self.do_prep=do_prep
        self.bestT=bestT
        self.cat_idx=cat_idx
        self.float_idx=float_idx
        self.impute_strategy=impute_strategy
        self.cv_strategy=cv_strategy
        BaseHelper.__init__(self)
    def get_pipe(self):
        if self.cv_strategy:
            inner_cv=regressor_q_stratified_cv(n_splits=10,n_repeats=3, strategy=self.cv_strategy,random_state=0,groupcount=self.groupcount)
        
        else:
            inner_cv=RepeatedKFold(n_splits=10, n_repeats=3, random_state=0)
        param_grid={}#'max_depth':list(range(2,5)),
                    #'n_estimators':[100,500,1000]
                    #   }
            
        steps=[('reg',GridSearchCV(GradientBoostingRegressor(random_state=0),param_grid=param_grid,cv=inner_cv))]
        if self.bestT:
            steps.insert(0,'xtransform',columnBestTransformer(float_k=len(self.float_idx)))
        outerpipe= Pipeline(steps=steps)
        if self.do_prep:
            steps=[('prep',missingValHandler(strategy=self.impute_strategy,cat_idx=self.cat_idx)),
                   ('post',outerpipe)]
            outerpipe=Pipeline(steps=steps)
        return outerpipe
        
class HGBR(BaseEstimator,RegressorMixin,myLogger,BaseHelper):
    def __init__(self,do_prep=True,cat_idx=None,float_idx=None):
        myLogger.__init__(self,name='HGBR.log')
        self.logger.info('starting histogram_gradient_boosting_reg logger')
        self.do_prep=do_prep
        self.cat_idx=cat_idx
        self.float_idx=float_idx
        BaseHelper.__init__(self)
    def get_pipe(self):
        steps=[
            ('reg',HistGradientBoostingRegressor())
        ]
        outerpipe= Pipeline(steps=steps)
        if self.do_prep:
            steps=[('prep',missingValHandler(strategy='pass-through',cat_idx=self.cat_idx)),
                   ('post',outerpipe)]
            outerpipe=Pipeline(steps=steps)
        return outerpipe


class ENet(BaseEstimator,RegressorMixin,myLogger,BaseHelper):
    def __init__(self,do_prep=True,impute_strategy='impute_knn5',
                 gridpoints=4,cv_strategy=None,groupcount=None,
                 float_idx=None,cat_idx=None,bestT=False):
        myLogger.__init__(self,name='enet.log')
        self.logger.info('starting enet logger')
        self.do_prep=do_prep
        self.gridpoints=gridpoints
        self.cv_strategy=cv_strategy
        self.groupcount=groupcount
        self.float_idx=float_idx
        self.cat_idx=cat_idx
        self.bestT=bestT
        self.impute_strategy=impute_strategy
        BaseHelper.__init__(self)

    def get_pipe(self,):
        if self.cv_strategy:
            inner_cv=regressor_q_stratified_cv(n_splits=10,n_repeats=3, strategy=self.cv_strategy,random_state=0,groupcount=self.groupcount)
        
        else:
            inner_cv=RepeatedKFold(n_splits=10, n_repeats=2, random_state=0)
        gridpoints=self.gridpoints
        param_grid={'l1_ratio':np.logspace(-2,-.03,gridpoints)}
        steps=[
            ('scaler',StandardScaler()),
            ('reg',GridSearchCV(ElasticNetCV(cv=inner_cv,normalize=False),param_grid=param_grid))]
            #('reg',ElasticNetCV(cv=inner_cv,normalize=True))]
            
        if self.bestT:
            steps.insert(0,('xtransform',columnBestTransformer(float_k=len(self.float_idx))))
        outerpipe=Pipeline(steps=steps)
        if self.do_prep:
            steps=[('prep',missingValHandler(strategy=self.impute_strategy,cat_idx=self.cat_idx)),
                   ('post',outerpipe)]
            outerpipe=Pipeline(steps=steps)
        return outerpipe

class RBFSVR(BaseEstimator,RegressorMixin,myLogger,BaseHelper):
    def __init__(self,do_prep=True,impute_strategy='impute_knn5',
                 gridpoints=4,cv_strategy=None,groupcount=None,
                 float_idx=None,cat_idx=None,bestT=False):
        myLogger.__init__(self,name='LinRegSupreme.log')
        self.logger.info('starting LinRegSupreme logger')
        self.do_prep=do_prep
        self.gridpoints=gridpoints
        self.cv_strategy=cv_strategy
        self.groupcount=groupcount
        self.bestT=bestT
        self.cat_idx=cat_idx
        self.float_idx=float_idx
        self.impute_strategy=impute_strategy
        BaseHelper.__init__(self)
    
    def get_pipe(self,):
        if self.cv_strategy:
            inner_cv=regressor_q_stratified_cv(n_splits=10,n_repeats=3, strategy=self.cv_strategy,random_state=0,groupcount=self.groupcount)
        
        else:
            inner_cv=RepeatedKFold(n_splits=10, n_repeats=3, random_state=0)
        gridpoints=self.gridpoints
        param_grid={'C':np.logspace(-2,2,gridpoints),
                   'gamma':np.logspace(-2,0.5,gridpoints)}
        steps=[
            ('scaler',StandardScaler()),
            ('reg',GridSearchCV(SVR(kernel='rbf',cache_size=10000,tol=1e-4,max_iter=5000),param_grid=param_grid))]
        if self.bestT:
            steps.insert(0,('xtransform',columnBestTransformer(float_k=len(self.float_idx))))
        outerpipe=Pipeline(steps=steps)
        if self.do_prep:
            steps=[('prep',missingValHandler(strategy=self.impute_strategy,cat_idx=self.cat_idx)),
                   ('post',outerpipe)]
            outerpipe=Pipeline(steps=steps)
        return outerpipe



class LinSVR(BaseEstimator,RegressorMixin,myLogger,BaseHelper):
    def __init__(self,do_prep=True,impute_strategy='impute_knn5',
                 gridpoints=4,cv_strategy=None,groupcount=None,
                 bestT=False,cat_idx=None,float_idx=None):
        myLogger.__init__(self,name='LinRegSupreme.log')
        self.logger.info('starting LinRegSupreme logger')
        self.do_prep=do_prep
        self.gridpoints=gridpoints
        self.cv_strategy=cv_strategy
        self.groupcount=groupcount
        self.bestT=bestT
        self.cat_idx=cat_idx
        self.float_idx=float_idx
        self.impute_strategy=impute_strategy
        BaseHelper.__init__(self)
        
    
    def get_pipe(self,):
        if self.cv_strategy:
            inner_cv=regressor_q_stratified_cv(n_splits=10,n_repeats=3, strategy=self.cv_strategy,random_state=0,groupcount=self.groupcount)
        
        else:
            inner_cv=RepeatedKFold(n_splits=10, n_repeats=3, random_state=0)
        gridpoints=self.gridpoints
        param_grid={'C':np.logspace(-2,4,gridpoints)}
        steps=[
            #('shrink_k1',shrinkBigKTransformer(selector=LassoLarsCV(cv=inner_cv,max_iter=32))), # retain a subset of the best original variables
            ('polyfeat',PolynomialFeatures(interaction_only=0,degree=2)), # create interactions among them
            
            ('drop_constant',dropConst()),
            ('shrink_k2',shrinkBigKTransformer(selector=LassoLarsCV(cv=inner_cv,max_iter=64))),
            ('scaler',StandardScaler()),
            ('reg',GridSearchCV(LinearSVR(random_state=0,tol=1e-4,max_iter=1000),param_grid=param_grid))]
        if self.bestT:
            steps=[steps[0],('xtransform',columnBestTransformer(float_k=len(self.float_idx))),*steps[1:]]
        outerpipe=Pipeline(steps=steps)
        if self.do_prep:
            steps=[('prep',missingValHandler(strategy=self.impute_strategy,cat_idx=self.cat_idx)),
                   ('post',outerpipe)]
            outerpipe=Pipeline(steps=steps)
        return outerpipe

        
class LinRegSupreme(BaseEstimator,RegressorMixin,myLogger,BaseHelper):
    def __init__(self,do_prep=True,impute_strategy='impute_knn5',
                 gridpoints=4,cv_strategy=None,groupcount=None,
                 bestT=False,cat_idx=None,float_idx=None):
        myLogger.__init__(self,name='LinRegSupreme.log')
        self.logger.info('starting LinRegSupreme logger')
        self.do_prep=do_prep
        self.gridpoints=gridpoints
        self.cv_strategy=cv_strategy
        self.groupcount=groupcount
        self.bestT=bestT
        self.cat_idx=cat_idx
        self.float_idx=float_idx
        self.impute_strategy=impute_strategy
        BaseHelper.__init__(self)
    
    
    def get_pipe(self,):
        if self.cv_strategy:
            inner_cv=regressor_q_stratified_cv(n_splits=10,n_repeats=1, strategy=self.cv_strategy,random_state=0,groupcount=self.groupcount)
        
        else:
            inner_cv=RepeatedKFold(n_splits=10, n_repeats=1, random_state=0)
        gridpoints=self.gridpoints
        transformer_list=[none_T(),log_T(),logp1_T()]#,logp1_T()] # log_T()]#
        steps=[
            ('shrink_k1',shrinkBigKTransformer(selector=LassoLarsCV(cv=inner_cv,max_iter=32))), # retain a subset of the best original variables
            ('polyfeat',PolynomialFeatures(interaction_only=0,degree=2)), # create interactions among them
            
            ('drop_constant',dropConst()),
            ('shrink_k2',shrinkBigKTransformer(selector=LassoLarsCV(cv=inner_cv,max_iter=64))), # pick from all of those options
            ('reg',LinearRegression())]
        if self.bestT:
            steps.insert(0,('xtransform',columnBestTransformer(float_k=len(self.float_idx))))

        X_T_pipe=Pipeline(steps=steps)
        #inner_cv=regressor_stratified_cv(n_splits=10,n_repeats=2,shuffle=True)
        


        Y_T_X_T_pipe=Pipeline(steps=[('ttr',TransformedTargetRegressor(regressor=X_T_pipe))])
        Y_T__param_grid={
            'ttr__transformer':transformer_list,
            'ttr__regressor__polyfeat__degree':[2],
        }
        outerpipe= GridSearchCV(Y_T_X_T_pipe,param_grid=Y_T__param_grid,cv=inner_cv)
        if self.do_prep:
            steps=[('prep',missingValHandler(strategy=self.impute_strategy,cat_idx=self.cat_idx)),
                   ('post',outerpipe)]
            outerpipe=Pipeline(steps=steps)
        
        return outerpipe
    
class NullModel(BaseEstimator,RegressorMixin):
    def __init__(self):
        pass
    def fit(self,x,y,w=None):
        pass
    def predict(self,x,):
        if len(x.shape)>1:
            return np.mean(x,axis=1)
        return x
 
class MultiPipe(BaseEstimator,RegressorMixin,myLogger,BaseHelper):
    def __init__(self,pipelist=[GBR],kwargs_list={},impute_strategy='impute_knn5',cat_idx=None):
        myLogger.__init__(self,name='multipipe.log')
        self.pipelist=pipelist
        self.kwargs_list=kwargs_list
        self.cat_idx=cat_idx
        self.impute_strategy=impute_strategy
        
    def get_pipe(self):
        try:
            pipe_n=len(self.pipelist)
            if type(self.kwargs_list) is dict:
                self.kwargs_list=[self.kwargs_list]*pipe_n
            #for kwargs in self.kwargs_list:
            #    kwargs['do_prep']=False
            est_pipes=[(p[0],p[1](**self.kwargs_list[i])) for i,p in enumerate(self.pipelist)]
            steps=[
                ('prep',missingValHandler(
                    strategy=self.impute_strategy,cat_idx=self.cat_idx)),
                ('post',StackingRegressor(est_pipes,passthrough=True,final_estimator=NullModel()))]   
            return Pipeline(steps=steps)
        except:
            self.logger.exception(f'error')
            assert False,'halt'
    
    def get_names(self):
        self.est
    
    #def extract_pipe
        
    
    
    
if __name__=="__main__":
    X, y= make_regression(n_samples=300,n_features=5,noise=1)
    lrs=LinRegSupreme(do_prep=True)
    lrs.fit(X,y)
    s=lrs.score(X,y)
    print(f'r2 score: {s}')
    cv=cross_validate(lrs,X,y,scoring='r2',cv=2)
    print(cv)
    print(cv['test_score'])