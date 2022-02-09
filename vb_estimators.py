import numpy as np
from scipy.optimize import least_squares,minimize
from sklearn.base import BaseEstimator, TransformerMixin,RegressorMixin
from sklearn.datasets import make_regression
from sklearn.pipeline import make_pipeline,Pipeline

from sklearn.ensemble import GradientBoostingRegressor,HistGradientBoostingRegressor,StackingRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Lars,Lasso,LassoCV,LassoLarsCV,ElasticNetCV,TweedieRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.preprocessing import StandardScaler, FunctionTransformer, PolynomialFeatures, OneHotEncoder, PowerTransformer
from sklearn.model_selection import cross_validate, train_test_split, RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import TransformedTargetRegressor
import matplotlib.pyplot as plt
from vb_transformers import shrinkBigKTransformer,logminus_T,exp_T,logminplus1_T,none_T,log_T, logp1_T,dropConst,columnBestTransformer
from missing_val_transformer import missingValHandler
#NOT implemented: from nonlinear_stacker import stackNonLinearTransforms
import os
import pandas as pd
from vb_cross_validator import regressor_q_stratified_cv
import logging, logging.handlers
try:
    import daal4py.sklearn
    daal4py.sklearn.patch_sklearn()
except:
    pass
    #print('no daal4py')
try: #for sklearn version 0.23, 0.24 for example
    from sklearn.experimental import enable_hist_gradient_boosting  
except:
    pass

class myLogger:
    def __init__(self,name=None):
        if name is None:
            name='vb_estimators.log'
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

class BaseHelper: #wrapper/helper function for pipelines; provides lots of flexibility for pipeline fitting; for example,
    #would allow you to create weights for each observation and potentially pass those to fit/transform/score functions
    def __init__(self): #trivial instantiation
        pass
    def fit(self,X,y):
        self.n_,self.k_=X.shape #dimensions of the training data
        
        self.pipe_=self.get_pipe()  #creates a copy of the pipeline that's ready to be fit
        try: #try/except for debugging purposes for capturing errors in fitting a pipeline
            self.pipe_.fit(X,y)
        except:
            self.logger.exception(f'error fitting pipeline')
            assert False,'halt fit'
        return self
    #three functions integral to every constructed pipeline
    def transform(self,X,y=None): #may never be used? INVESTIGATE
        return self.pipe_.transform(X,y)
    def score(self,X,y):
        return self.pipe_.score(X,y)
    def predict(self,X):
        return self.pipe_.predict(X)
    
    def extractParams(self,param_dict,prefix=''): #splits parameters into static and tunable hyperparameters
        hyper_param_dict={}
        static_param_dict={}
        for param_name,val in param_dict.items():
            if type(val) is list:
                hyper_param_dict[prefix+param_name]=val
            else:
                static_param_dict[param_name]=val
        return hyper_param_dict,static_param_dict

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
        return np.nan_to_num(y,nan=1e298)
    
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
    
    def pipe_sq_residuals(self,B,X,y):
        return np.sum((self.pipe_(B,X)-y)**2)
    
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
            self.fit_est_=minimize(self.pipe_sq_residuals, np.ones(self.k),args=(X, y),method='BFGS')# 
        else:
            self.fit_est_=least_squares(self.pipe_residuals, np.ones(self.k),args=(X, y))# 
        return self
    
    def predict(self,X):
        B=self.fit_est_.x #.x pulls out the fitted coefficients selected by the optimizer
        return self.pipe_(B,X)

class FlexiblePipe(BaseEstimator,RegressorMixin,myLogger,BaseHelper):
    def __init__(
        self,do_prep=True,functional_form_search =False,
        prep_dict={'impute_strategy':'impute_knn5'},gridpoints=4,
        inner_cv=None,groupcount=None,bestT=False,
        cat_idx=None,float_idx=None,flex_kwargs={}
        ):
        
        myLogger.__init__(self,name='l1lars.log')
        self.logger.info('starting l1lars logger')
        self.do_prep=do_prep
        self.functional_form_search=functional_form_search #if True, functional_form is tuned as a hyperparameter
        self.gridpoints=gridpoints
        self.inner_cv=inner_cv
        self.groupcount=groupcount
        self.bestT=bestT
        self.cat_idx=cat_idx
        self.float_idx=float_idx
        self.prep_dict=prep_dict
        self.flex_kwargs=flex_kwargs #these are passed directly to the FlexibleEstimator
        BaseHelper.__init__(self)
        
    def get_pipe(self,):
        if self.inner_cv is None:
            inner_cv=RepeatedKFold(n_splits=10, n_repeats=1, random_state=0)
        else:
            inner_cv=self.inner_cv
            
        steps=[
            ('scaler',StandardScaler()),
            ('select',shrinkBigKTransformer(max_k=8)),
            ('reg',FlexibleEstimator(**self.flex_kwargs))
        ]
        if self.bestT:
            steps.insert(0,'xtransform',columnBestTransformer(float_k=len(self.float_idx)))

        pipe=Pipeline(steps=steps)
        # selecting features. k_share allows you to choose the fraction of features that are retained using the LARS algorithm
        param_grid={'select__k_share':np.linspace(0.2,1,self.gridpoints*2)}
        if self.functional_form_search:
            param_grid['reg__form']=['powXB','expXB']#,'linear']

        outerpipe=GridSearchCV(pipe,param_grid=param_grid)
        if self.do_prep:
            steps=[('prep',missingValHandler(prep_dict=self.prep_dict)),
                   ('post',outerpipe)]
            outerpipe=Pipeline(steps=steps)
            
        return outerpipe

class FlexibleGLM(BaseEstimator,RegressorMixin,myLogger,BaseHelper):
    def __init__(
        self,do_prep=True,prep_dict={'impute_strategy':'impute_knn5'},inner_cv=None,bestT=False,
        cat_idx=None,float_idx=None,est_kwargs=None,gridpoints=5
    ):
        myLogger.__init__(self,name='gbr.log')
        self.logger.info('starting flexible GLM logger')
        self.do_prep=do_prep
        self.bestT=bestT
        self.cat_idx=cat_idx
        self.float_idx=float_idx
        self.prep_dict=prep_dict
        self.inner_cv=inner_cv
        self.est_kwargs=est_kwargs
        self.gridpoints=gridpoints
        BaseHelper.__init__(self)
        
    def get_pipe(self):
        try:
            
            if self.inner_cv is None:
                inner_cv=RepeatedKFold(n_splits=10, n_repeats=1, random_state=0)
            else:
                inner_cv=self.inner_cv
            if self.est_kwargs is None:
                self.est_kwargs={
                    'reg__alpha':np.logspace(-5,10,self.gridpoints).tolist(), #investigate the ideal range for alpha
                    'reg__power':[0,*np.logspace(1,3,self.gridpoints-1).tolist()], #investigate power values between 2 and 3
                    'select__max_k':[4,8,32]} #maybe look to tweak using k_share
            steps=[
                ('scaler',StandardScaler()),
                ('select',shrinkBigKTransformer(max_k=8)),
                ('reg',TweedieRegressor())]
            if self.bestT:
                steps.insert(0,'xtransform',columnBestTransformer(float_k=len(self.float_idx)))

            outerpipe=GridSearchCV(Pipeline(steps=steps),param_grid=self.est_kwargs,cv=inner_cv)
            if self.do_prep:
                steps=[('prep',missingValHandler(prep_dict=self.prep_dict)),
                       ('post',outerpipe)]
                outerpipe=Pipeline(steps=steps)
            return outerpipe
        except:
            self.logger.exception(f'get_pipe error for flexibleGLM')

class L1Lars(BaseEstimator,RegressorMixin,myLogger,BaseHelper): #Regularized regression via LASSO(L1)/LARS; note no
    #est_kwargs due to relative simplicity of this estimator, but this feature could potentially be added
    def __init__(self,do_prep=True,prep_dict={'impute_strategy':'impute_knn5'},
                 max_n_alphas=1000,inner_cv=None,groupcount=None, #max_n_alphas coming from GUI
                 bestT=False,cat_idx=None,float_idx=None):
        myLogger.__init__(self,name='l1lars.log')
        self.logger.info('starting l1lars logger')
        self.do_prep=do_prep
        self.max_n_alphas=max_n_alphas
        self.inner_cv=inner_cv
        self.groupcount=groupcount #not used, can be deleted
        self.bestT=bestT
        self.cat_idx=cat_idx
        self.float_idx=float_idx
        self.prep_dict=prep_dict
        #self.pipe_=self.get_pipe() #formerly inside basehelper         
        BaseHelper.__init__(self)
    
    def get_pipe(self,):
        if self.inner_cv is None:
            inner_cv=RepeatedKFold(n_splits=10, n_repeats=1, random_state=0)
        else:
            inner_cv=self.inner_cv
        
        steps=[('scaler',StandardScaler()), #standardizes the X values
            ('reg',LassoLarsCV(cv=inner_cv,max_n_alphas=self.max_n_alphas,normalize=False))]
        if self.bestT:
            steps.insert(0,'xtransform',columnBestTransformer(float_k=len(self.float_idx)))
        outerpipe=Pipeline(steps=steps)
        
        if self.do_prep:
            steps=[('prep',missingValHandler(prep_dict=self.prep_dict)),
                   ('post',outerpipe)]
            outerpipe=Pipeline(steps=steps)
        return outerpipe

class GBR(BaseEstimator,RegressorMixin,myLogger,BaseHelper): #building a GBR object
    def __init__(self,do_prep=True,prep_dict={'impute_strategy':'impute_knn5'},inner_cv=None,bestT=False,cat_idx=None,float_idx=None,est_kwargs=None):
        myLogger.__init__(self,name='gbr.log')
        self.logger.info('starting gradient_boosting_reg logger')
        self.do_prep=do_prep #whether or not to do data prep (binarize/impute)
        self.bestT=bestT #searches for best transformation of each feature for prediction
        self.cat_idx=cat_idx
        self.float_idx=float_idx
        self.prep_dict=prep_dict
        self.inner_cv=inner_cv #suppling the cross-validator
        self.est_kwargs=est_kwargs
        BaseHelper.__init__(self) #initializing the BaseHelper class to inherit from it

    def get_pipe(self): #getting a GBR pipeline; this code follows a template shared by other pipeline classes
        if self.inner_cv is None:
            inner_cv=RepeatedKFold(n_splits=10, n_repeats=1, random_state=0) #default inner_cv iterator
        else:
            inner_cv=self.inner_cv
        if self.est_kwargs is None: #non-hard-coded key word arguments for the estimator supplied at the end of the pipeline
            self.est_kwargs={'max_depth':[3,4],'n_estimators':[64,128]}
        hyper_param_dict,gbr_params=self.extractParams(self.est_kwargs) #pulling out tunable hyperparameters that will be selected
        #by grid search and static parameters that will be passed directly to the estimator
        if not 'random_state' in gbr_params:
            gbr_params['random_state']=0
        steps=[('reg',GridSearchCV(GradientBoostingRegressor(**gbr_params),param_grid=hyper_param_dict,cv=inner_cv))]
        #** unpacks a dictionary and provides its elements to the function;
        if self.bestT: #user asking for best feature transforms using the GUI
            steps.insert(0,'xtransform',columnBestTransformer(float_k=len(self.float_idx))) #GET BACK TO columnBestTransformer
            #in Doug's experience, columnBestTransformer overbuilds the pipeline and doesn't generalize well
        outerpipe=Pipeline(steps=steps)
        if self.do_prep: #wrapping outerpipe inside another outerpipe pipeline
            steps=[('prep',missingValHandler(prep_dict=self.prep_dict)),
                   ('post',outerpipe)]
            outerpipe=Pipeline(steps=steps)
        return outerpipe
        
class HGBR(BaseEstimator,RegressorMixin,myLogger,BaseHelper): #Histogram gradient boosting class; note no
    #est_kwargs, but this feature could potentially be added
    def __init__(self,do_prep=True,prep_dict=None):
        myLogger.__init__(self,name='HGBR.log')
        self.logger.info('starting histogram_gradient_boosting_reg logger')
        self.do_prep=do_prep
        self.prep_dict=prep_dict
        #self.pipe_=self.get_pipe() #formerly inside basehelper         
        BaseHelper.__init__(self)
    def get_pipe(self):
        steps=[
            ('reg',HistGradientBoostingRegressor())
        ]
        outerpipe= Pipeline(steps=steps)
        if self.do_prep:
            steps=[('prep',missingValHandler(prep_dict=dict(impute_strategy='pass-through',cat_idx=self.prep_dict['cat_idx']))),
                   ('post',outerpipe)]
            outerpipe=Pipeline(steps=steps)
        return outerpipe

class ENet(BaseEstimator,RegressorMixin,myLogger,BaseHelper):#Elastic Net class; note no
    #est_kwargs due to simplicity of this estimator, but this feature could potentially be added
    def __init__(self,do_prep=True,prep_dict={'impute_strategy':'impute_knn5'},
                 gridpoints=4,inner_cv=None,groupcount=None, #gridpoints coming from GUI
                 float_idx=None,cat_idx=None,bestT=False):
        myLogger.__init__(self,name='enet.log')
        self.logger.info('starting enet logger')
        self.do_prep=do_prep
        self.gridpoints=gridpoints
        self.inner_cv=inner_cv
        self.groupcount=groupcount
        self.float_idx=float_idx
        self.cat_idx=cat_idx
        self.bestT=bestT
        self.prep_dict=prep_dict
        #self.pipe_=self.get_pipe() #formerly inside basehelper         
        BaseHelper.__init__(self)

    def get_pipe(self,):
        if self.inner_cv is None:
            inner_cv=RepeatedKFold(n_splits=10, n_repeats=1, random_state=0)
        else:
            inner_cv=self.inner_cv
        gridpoints=self.gridpoints
        #param_grid={'l1_ratio':1-np.logspace(-2,-.03,gridpoints)} #manually creating a list of gridpoints for the l1_ratio;
        #ENet chooses these with its own internal GridSearchCV
        l1_ratio=1-np.logspace(-2,-.03,gridpoints*2) #regularization hyperparameter for ENet; creating a list of gridpoints;
        #multiplication by 2 somewhat arbitrary; Sci-kit learn documentation?
        n_alphas=gridpoints*5 #another regularization hyperparameter for ENet; multiplication by 5 chosen somewhat arbitrarily
        steps=[
            ('scaler',StandardScaler()), #StandardScaler chosen as perceived "best" option for scaling data (from Doug's experience)
            #('reg',GridSearchCV(ElasticNetCV(cv=inner_cv,normalize=False,),param_grid=param_grid))]; commented out to instead pass
            #list of values to l1_ratio instead of GridSearchCV
            ('reg',ElasticNetCV(cv=inner_cv,normalize=False,l1_ratio=l1_ratio,n_alphas=n_alphas))]
            
        if self.bestT:
            steps.insert(0,('xtransform',columnBestTransformer(float_k=len(self.float_idx))))
        outerpipe=Pipeline(steps=steps)
        if self.do_prep:
            steps=[('prep',missingValHandler(prep_dict=self.prep_dict)),
                   ('post',outerpipe)]
            outerpipe=Pipeline(steps=steps)
        return outerpipe

class RBFSVR(BaseEstimator,RegressorMixin,myLogger,BaseHelper):
    def __init__(self,do_prep=True,prep_dict={'impute_strategy':'impute_knn5'},
                 gridpoints=4,inner_cv=None,groupcount=None,
                 float_idx=None,cat_idx=None,bestT=False):
        myLogger.__init__(self,name='LinRegSupreme.log')
        self.logger.info('starting LinRegSupreme logger')
        self.do_prep=do_prep
        self.gridpoints=gridpoints
        self.inner_cv=inner_cv
        self.groupcount=groupcount
        self.bestT=bestT
        self.cat_idx=cat_idx
        self.float_idx=float_idx
        self.prep_dict=prep_dict
        #self.pipe_=self.get_pipe() #formerly inside basehelper         
        BaseHelper.__init__(self)
    
    def get_pipe(self,):
        if self.inner_cv is None:
            inner_cv=RepeatedKFold(n_splits=10, n_repeats=1, random_state=0)
        else:
            inner_cv=self.inner_cv
        # parameter grid for SVR using two parameters, C and gamma
        param_grid={
            'C':np.logspace(-2,2,self.gridpoints*2),
            'gamma':np.logspace(-2,0.5,self.gridpoints)}
        steps=[
            ('scaler',StandardScaler()),
            # TEST removal of cache_size, tol and max_iter; probably don't need these
            ('reg',GridSearchCV(SVR(kernel='rbf',cache_size=10000,tol=1e-4,max_iter=5000),param_grid=param_grid))]
        if self.bestT:
            steps.insert(0,('xtransform',columnBestTransformer(float_k=len(self.float_idx))))
        outerpipe=Pipeline(steps=steps)
        if self.do_prep:
            steps=[('prep',missingValHandler(prep_dict=self.prep_dict)),
                   ('post',outerpipe)]
            outerpipe=Pipeline(steps=steps)
        return outerpipe

class LinSVR(BaseEstimator,RegressorMixin,myLogger,BaseHelper):
    def __init__(self,do_prep=True,prep_dict={'impute_strategy':'impute_knn5'},
                 gridpoints=4,inner_cv=None,groupcount=None,
                 bestT=False,cat_idx=None,float_idx=None):
        myLogger.__init__(self,name='LinRegSupreme.log')
        self.logger.info('starting LinRegSupreme logger')
        self.do_prep=do_prep
        self.gridpoints=gridpoints
        self.inner_cv=inner_cv
        self.groupcount=groupcount
        self.bestT=bestT
        self.cat_idx=cat_idx
        self.float_idx=float_idx
        self.prep_dict=prep_dict
        #self.pipe_=self.get_pipe() #formerly inside basehelper         
        BaseHelper.__init__(self)

    def get_pipe(self,):
        if self.inner_cv is None:
            inner_cv=RepeatedKFold(n_splits=10, n_repeats=1, random_state=0)
        else:
            inner_cv=self.inner_cv
            
        gridpoints=self.gridpoints
        param_grid={'C':np.logspace(-2,4,gridpoints*4)}
        steps=[
            #('shrink_k1',shrinkBigKTransformer(selector=LassoLarsCV(cv=inner_cv,max_iter=32))), # retain only a subset of the
            # original variables for continued use
            ('polyfeat',PolynomialFeatures(interaction_only=False,degree=2)), # create every 2nd-order interaction among the features
            # including squared terms
            ('drop_constant',dropConst()), #drops features without variance, including created interactions
            ('shrink_k2',shrinkBigKTransformer(selector=LassoLarsCV(cv=inner_cv,max_iter=64))),
            ('scaler',StandardScaler()),
            # TEST removal of cache_size, tol and max_iter; probably don't need these
            ('reg',GridSearchCV(LinearSVR(random_state=0,tol=1e-4,max_iter=1000),param_grid=param_grid))]
        if self.bestT:
            steps=[steps[0],('xtransform',columnBestTransformer(float_k=len(self.float_idx))),*steps[1:]]
        outerpipe=Pipeline(steps=steps)
        if self.do_prep:
            steps=[('prep',missingValHandler(prep_dict=self.prep_dict)),
                   ('post',outerpipe)]
            outerpipe=Pipeline(steps=steps)
        return outerpipe

class LinRegSupreme(BaseEstimator,RegressorMixin,myLogger,BaseHelper):
    def __init__(self,do_prep=True,prep_dict={'impute_strategy':'impute_knn5'},
                 gridpoints=4,inner_cv=None,groupcount=None,
                 bestT=False,cat_idx=None,float_idx=None):
        myLogger.__init__(self,name='LinRegSupreme.log')
        self.logger.info('starting LinRegSupreme logger')
        self.do_prep=do_prep
        self.gridpoints=gridpoints
        self.inner_cv=inner_cv
        self.groupcount=groupcount
        self.bestT=bestT
        self.cat_idx=cat_idx
        self.float_idx=float_idx
        self.prep_dict=prep_dict
        #self.pipe_=self.get_pipe() #formerly inside basehelper         
        BaseHelper.__init__(self)

    def get_pipe(self,):
        if self.inner_cv is None:
            inner_cv=RepeatedKFold(n_splits=10, n_repeats=1, random_state=0)
        else:
            inner_cv=self.inner_cv

        # gridpoints=self.gridpoints
        transformer_list=[none_T(),log_T(),logp1_T()]# Using 3 of many options here: none_T,logp1_T(),log_T()
        steps=[
            ('shrink_k1',shrinkBigKTransformer(selector=LassoLarsCV(cv=inner_cv,max_iter=32))), # retain a subset of the best original variables
            ('polyfeat',PolynomialFeatures(interaction_only=0,degree=2)), # create interactions among them
            ('drop_constant',dropConst()),
            ('shrink_k2',shrinkBigKTransformer(selector=LassoLarsCV(cv=inner_cv,max_iter=64))), # pick from all of those options
            ('reg',LinearRegression())]
        if self.bestT:
            steps.insert(0,('xtransform',columnBestTransformer(float_k=len(self.float_idx))))

        X_T_pipe=Pipeline(steps=steps)
        #develop a new pipeline that allows transformation of y in addition to X, which other scikit learn transformers don't
        Y_T_X_T_pipe=Pipeline(steps=[('ttr',TransformedTargetRegressor(regressor=X_T_pipe))])
        Y_T__param_grid={
            'ttr__transformer':transformer_list,
            'ttr__regressor__polyfeat__degree':[2], #could use other degrees here if desired
        }
        outerpipe= GridSearchCV(Y_T_X_T_pipe,param_grid=Y_T__param_grid,cv=inner_cv)
        if self.do_prep:
            steps=[('prep',missingValHandler(prep_dict=self.prep_dict)),
                   ('post',outerpipe)]
            outerpipe=Pipeline(steps=steps)
        
        return outerpipe

#this class doesn't appear to be used in the project
class NullModel(BaseEstimator,RegressorMixin):
    def __init__(self):
        pass
    def fit(self,x,y,w=None): # w is weights
        pass
    def predict(self,x,):
        if len(x.shape)>1:
            return np.mean(x,axis=1)
        return x
 
class MultiPipe(BaseEstimator,RegressorMixin,myLogger,BaseHelper):
    def __init__(self,pipelist=None,prep_dict=None,stacker_estimator=None):
        myLogger.__init__(self,name='multipipe.log')
        self.pipelist=pipelist
        self.prep_dict=self.getPrepDict(prep_dict)
        self.stacker_estimator=stacker_estimator
        self.pipe_=self.get_pipe() #formerly inside basehelper
        BaseHelper.__init__(self)

    #TEST deletion of the following code
    def getPrepDict(self,prep_dict):
        if self.pipelist is None:
            print('empty MultiPipe!')
            return None
        if prep_dict is None:
            for pname,pdict in self.pipelist:
                if 'prep_dict' in pdict['pipe_kwargs']:
                    return pdict['pipe_kwargs']['prep_dict']
            assert False,f'no prep_dict found in any pipedict within self.pipelist:{self.pipelist}'
        else: return prep_dict

    def get_pipe(self):
        
        try:
            # calling the pipelines in self.pipelist with their keyword arguments
            est_pipes=[(p[0],p[1]['pipe'](**p[1]['pipe_kwargs'])) for p in self.pipelist]
            final_e=self.stacker_estimator
            steps=[
                ('prep',missingValHandler(prep_dict=self.prep_dict)),
                #passthrough=True would add the original covariates to the final stacked regressor model in addition
                #to the y-hats of the component pipelines
                ('post',make_pipeline(StackingRegressor(est_pipes,passthrough=False,final_estimator=final_e,n_jobs=1)))]   
            return Pipeline(steps=steps)
        except:
            self.logger.exception(f'error')
            assert False,'halt'
    
    def get_pipe_names(self):
        pipe_names=[pipe_tup[0] for pipe_tup in self.pipelist]
        return pipe_names

    #pulls out stand-alone pipelines from the stacked regressor
    def get_individual_post_pipes(self,names=None):
        if names is None:
            names=self.get_pipe_names()
        if type(names) is str:
            names=[names]
        pipe_dict={} 
        for name in names:
            pipe_dict[name]=self.pipe_['post']['stackingregressor'].named_estimators_[name]
        return pipe_dict
    
    def get_prep(self):
        return self.pipe_['prep']
                
    def build_individual_fitted_pipelines(self,names=None):
        pipe_dict=self.get_individual_post_pipes(names=names)
        prep=self.get_prep()
        fitted_ipipe_dict={} #i used for individual
        for pname,pipe in pipe_dict.items():
            fitted_steps=[('prep',prep),('post',pipe)]
            fitted_ipipe_dict[pname]=FCombo(fitted_steps) #calling the Frankenstein fitted combo
        return fitted_ipipe_dict

class FCombo(BaseEstimator,RegressorMixin,myLogger):
    #this is built by MultiPipe and allows for re-use of the data prep step in that function
    #typical use of FCombo is to create stand alone pipelines from the stacking regressor component pipelines; no additional
    #fitting is required, as it has already happened when the stacking regressor was fit
    def __init__(self,fitted_steps):
        myLogger.__init__(self)
        self.fitted_steps=fitted_steps

    def fit(self,X,y):
        assert False,'fit called! this is a fitted combo!'

    #score is a measure of model performance; higher is better
    #def score(self,X,y):
    #    return .score(X,y)

    def predict(self,X):
        step_n=len(self.fitted_steps)
        # step_names=[step_tup[0] for step_tup in self.fitted_steps]
        Xt=X.copy()
        #Steps 1 to n-1 are transformers, step n is an estimator
        for s in range(0,step_n-1):
            Xt=self.fitted_steps[s][1].transform(Xt)
        return self.fitted_steps[-1][1].predict(Xt)

#this code would be run if you ran the python file directly from a command line
if __name__=="__main__":
    X, y= make_regression(n_samples=300,n_features=5,noise=1)
    lrs=LinRegSupreme(do_prep=True)
    lrs.fit(X,y)
    s=lrs.score(X,y)
    print(f'r2 score: {s}')
    cv=cross_validate(lrs,X,y,scoring='r2',cv=2)
    print(cv)
    print(cv['test_score'])