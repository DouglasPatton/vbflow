import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import ElasticNetCV, LinearRegression, Lars
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer

class VBHelper:
    def __init__(self,test_share,cv_folds,cv_reps,cv_count,rs):
        self.test_share=test_share
        self.cv_folds=cv_folds
        self.cv_reps=cv_reps
        self.cv_count=cv_count
        self.rs=rs
        
        # below are added in the notebook
        self.scorer_list=None
        self.max_k=None
        self.estimator_dict=None

        
    
    def plotCVScores(self,cv_score_dict,sort=1):
        colors = ['r', 'g', 'b', 'm', 'c', 'y', 'k']    
        fig=plt.figure(figsize=[12,15])
        plt.suptitle(f"Model Scores Across {self.cv_count} Cross Validation Runs. ")
        s_count=len(self.scorer_list)
        xidx=np.arange(self.cv_count) # place holder for scatterplot

        for s_idx, scorer in enumerate(self.scorer_list):
            ax=fig.add_subplot(f'{s_count}1{s_idx}')
            #ax.set_xlabel('estimator')
            #ax.set_ylabel(scorer)
            ax.set_title(scorer)
            for e_idx,estimator_name in enumerate(cv_score_dict.keys()):
                scores=cv_score_dict[estimator_name][scorer]
                if sort: scores.sort()
                ax.plot(xidx,scores,color=colors[e_idx],alpha=0.5,label=estimator_name+' cv score='+str(np.mean(cv_score_dict[estimator_name][scorer])))
                #ax.hist(scores,density=1,color=colors[e_idx],alpha=0.5,label=estimator_name+' cv score='+str(np.mean(cv_score_dict[estimator_name][scorer])))
            ax.grid(True)
            ax.xaxis.set_ticks([])
            ax.xaxis.set_visible(False)
            ax.legend(loc=4)
            #fig.show()
            
            

class shrinkBigKTransformer(BaseEstimator,TransformerMixin):
    def __init__(self,max_k=500,selector=None):
        self.max_k=max_k
        if selector is None:
            self.selector='Lars' 
        else: self.selector=selector
            
        
    def fit(self,X,y):
        assert not y is None,f'y:{y}'
        if self.selector=='Lars':
            selector=Lars(fit_intercept=1,normalize=1,n_nonzero_coefs=self.max_k)
        elif self.selector=='elastic-net':
            selector=ElasticNetCV()
        else:
            selector=self.selector
        k=X.shape[1]
        selector.fit(X,y)
        self.col_select_=np.arange(k)[np.abs(selector.coef_)>0.0001]
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
        pass
    def fit(self,X,y=None):
        xmin=X.min()
        if xmin<0:
            self.min_shift_=-xmin
        else:
            self.min_shift_=0
        return self
    
    def transform(self,X,y=None):
        X[X<-self.min_shift_]=-self.min_shift_ # added to avoid np.log(neg)
        return np.log1p(X+self.min_shift_)
        
    def inverse_transform(self,X,y=None):
        return np.expm1(X)-self.min_shift_
    
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
    
class none_T(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        return X
    def inverse_transform(self,X):
        return X  
    
        
    