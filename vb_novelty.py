import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin,RegressorMixin
from sklearn.datasets import make_regression
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.svm import OneClassSVM
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler

from missing_val_transformer import missingValHandler
from vb_estimators import BaseHelper,myLogger
import matplotlib.pyplot as plt


class NoveltyPipe(BaseEstimator,RegressorMixin,myLogger,BaseHelper):
    def __init__(self,alpha=0.05,do_prep=True,prep_dict={'impute_strategy':'impute_knn5'}):
        self.alpha=alpha
        self.do_prep=do_prep
        self.prep_dict=prep_dict
        BaseHelper.__init__(self)
        myLogger.__init__(self,name='novelty_pipe.log')
    
    def get_pipe(self):
        steps=[
            ('scaler',StandardScaler()),
            ('reg',OneClassSVM(nu=self.alpha))
        ]

        outerpipe=Pipeline(steps=steps)

        if self.do_prep:
            steps=[('prep',missingValHandler(prep_dict=self.prep_dict)),
                   ('post',outerpipe)]
            outerpipe=Pipeline(steps=steps)
            
        return outerpipe
    
class NoveltyAssess(myLogger):
    def __init__(self,inner_cv=None,alpha=0.01):
        self.inner_cv=inner_cv
        self.alpha=alpha
        myLogger.__init__(self,name='novelty_assessment.log')
        
    def fit(self,X,y=None):
        if type(X) is pd.DataFrame:
            X=X.to_numpy()
        if self.inner_cv is None:
            inner_cv=RepeatedKFold(n_splits=10, n_repeats=10, random_state=0)
        else:
            inner_cv=self.inner_cv
        self.pipe_list_=[]
        
        cv_count=inner_cv.get_n_splits(X)
        for cv_i,(train_idx,test_idx) in enumerate(inner_cv.split(X)):
            if cv_i==0:
                n_splits=-(-X.shape[0]//test_idx.shape[0]) #ceiling divide
                col_idx=0
                novelty_hat=np.zeros((X.shape[0],cv_count//n_splits),dtype=np.int8)
            else:
                if cv_i%n_splits==0:
                    col_idx+=1
            pipe=NoveltyPipe(alpha=self.alpha).fit(X[train_idx],y)
            novelty_hat[test_idx,col_idx]=pipe.predict(X[test_idx])
            self.pipe_list_.append(pipe)
        self.novelty_hat_=novelty_hat
        self.single_pipe_=NoveltyPipe(alpha=self.alpha).fit(X,y)
        return self
                                   
    def single_predict(self,X):
        if type(X) is pd.DataFrame:
            X=X.to_numpy()                           
        return self.single_pipe_.predict(X)
        
    def cv_predict(self,X):
        if type(X) is pd.DataFrame:
            X=X.to_numpy()
        m=len(self.pipe_list_)
        novelty_hat=np.zeros((X.shape[0],m))
        for i,pipe in enumerate(self.pipe_list_):
            novelty_hat[:,i]=pipe.predict(X)
        return novelty_hat.mean(axis=1)
                                   
    
    
                                   
if __name__=="__main__":
    X, y= make_regression(n_samples=300,n_features=2,noise=10)
    NA=NoveltyAssess().fit(X)
    print(NA.novelty_hat_.mean(axis=1))
    print(NA.cv_predict(X).mean())
    
    
                                   