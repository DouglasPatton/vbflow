from time import time
import logging
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, LinearRegression, Lars
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_validate, train_test_split, RepeatedKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer,KNNImputer

#import sys
#sys.path.append(os.path.abspath('..'))#sys.path[0] + '/..') 
from vb_cross_validator import regressor_q_stratified_cv

import logging,logging.handlers
class myLogger:
    def __init__(self,name=None):
        if name is None:
            name='vbflow.log'
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
        
class VBHelper:
    def __init__(self,test_share=0.2,cv_folds=5,cv_reps=2,random_state=0,cv_groupcount=None,cv_strategy=None):
        self.test_share=test_share
        self.cv_folds=cv_folds
        self.cv_reps=cv_reps
        self.cv_count=cv_reps*cv_folds
        self.cv_strategy=cv_strategy
        self.cv_groupcount=cv_groupcount
        self.rs=random_state
        
        # below are added in the notebook
        self.scorer_list=None
        self.max_k=None
        self.estimator_dict=None
        self.logger=logging.getLogger()
    
    def setData(self,X_df,y_df):
        self.X_df=X_df
        self.y_df=y_df
        self.cat_idx,self.cat_vars=zip(*[(i,var) for i,(var,dtype) in enumerate(dict(X_df.dtypes).items()) if dtype=='object'])
        self.float_idx=[i for i in range(X_df.shape[1]) if i not in self.cat_idx]
        print(self.cat_idx,self.cat_vars)

    def train_test_split(self):
        return train_test_split(
            self.X_df, self.y_df, 
            test_size=self.test_share,random_state=self.rs)
    
    
    
    
        
    def runCrossValidate(self,n_jobs=4):
        cv_results={}
        for estimator_name,model in self.model_dict.items():
            start=time()
            model_i=cross_validate(
                model, self.X_df, self.y_df, return_estimator=True, 
                scoring=self.scorer_list, cv=self.cv(), n_jobs=n_jobs)
            end=time()
            print(f"{estimator_name},{[(scorer,np.mean(model_i[f'test_{scorer}'])) for scorer in self.scorer_list]}, runtime:{(end-start)/60} min.")
            cv_results[estimator_name]=model_i
        self.cv_results=cv_results
        
    def cv(self,):
        if self.cv_strategy is None:
            return RepeatedKFold(
                n_splits=self.cv_folds, n_repeats=self.cv_reps, random_state=self.rs)
        else:
            if self.groupcount is None:
                self.groupcount=5
            return regressor_q_stratified_cv(
                n_splits=self.cv_folds, n_repeats=self.cv_reps, 
                random_state=self.rs,groupcount=self.cv_groupcount,strategy=self.cv_strategy)

    def predictCVYhat(self,):
        pass#for 
        
        
    def plotCVYhat(self,):
        for est_name,model_i in self.cv_results.items():
            pass#for 
    
    def buildCVScoreDict(self):
        cv_results=self.cv_results
        scorer_list=self.scorer_list
        cv_score_dict={}
        cv_score_dict_means={}
        for idx,(estimator_name,result) in enumerate(cv_results.items()):
            #cv_estimators=result['estimator']
            model_idx_scoredict={scorer:result[f'test_{scorer}'] for scorer in scorer_list}# fstring bc how cross_validate stores list of metrics
            cv_score_dict[estimator_name]=model_idx_scoredict 
            model_idx_mean_scores={scorer:np.mean(scores) for scorer,scores in model_idx_scoredict.items()}
            cv_score_dict_means[estimator_name]=model_idx_mean_scores
        self.cv_score_dict_means=cv_score_dict_means
        self.cv_score_dict=cv_score_dict
        
    def viewCVScoreDict(self):
        for scorer in self.scorer_list:
            print(f'scores for scorer: {scorer}:')
        for estimator_name in self.model_dict:
            print(f'    {estimator_name}:{self.cv_score_dict_means[estimator_name][scorer]}')
    
    def plotCVScores(self,sort=1):
        cv_score_dict=self.cv_score_dict
        colors = plt.get_cmap('tab20')(np.arange(20))#['r', 'g', 'b', 'm', 'c', 'y', 'k']    
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
            

    
        
    
