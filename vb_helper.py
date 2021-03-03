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
from vb_estimators import MultiPipe,FCombo

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
    def __init__(self,test_share=0.2,cv_folds=5,cv_reps=2,random_state=0,cv_strategy=None,run_stacked=True,cv_n_jobs=4):
        self.cv_n_jobs=cv_n_jobs
        self.run_stacked=run_stacked
        self.setProjectCVDict(cv_folds,cv_reps,cv_strategy)
        self.test_share=test_share
        self.rs=random_state
        #below attributes moved to self.project_cv_dict
        #self.cv_folds=cv_folds
        #self.cv_reps=cv_reps
        #self.cv_count=cv_reps*cv_folds
        #self.cv_strategy=cv_strategy
        #self.cv_groupcount=cv_groupcount
        
        
        # below are added in the notebook
        self.scorer_list=None
        self.max_k=None
        self.estimator_dict=None
        self.model_dict=None
        self.logger=logging.getLogger()
        
    def setProjectCVDict(self,cv_folds,cv_reps,cv_strategy):
        if cv_folds is None:
            cv_folds=10
        if cv_reps is None:
            cv_reps=1
        cv_count=cv_reps*cv_folds
        self.project_CV_dict={
            'cv_folds':cv_folds,
            'cv_reps':cv_reps,
            'cv_count': cv_count,
            'cv_strategy':cv_strategy
        }
        
    
    
    def setData(self,X_df,y_df):
        self.X_df=X_df
        self.y_df=y_df
        self.cat_idx,self.cat_vars=zip(*[(i,var) for i,(var,dtype) in enumerate(dict(X_df.dtypes).items()) if dtype=='object'])
        self.float_idx=[i for i in range(X_df.shape[1]) if i not in self.cat_idx]
        #print(self.cat_idx,self.cat_vars)

    def train_test_split(self):
        return train_test_split(
            self.X_df, self.y_df, 
            test_size=self.test_share,random_state=self.rs)
    
    def setEstimatorDict(self,estimator_dict):
        if self.run_stacked:
            self.estimator_dict={'multi_pipe':{'pipe':MultiPipe,'pipe_kwargs':{'pipelist':list(estimator_dict.items())}}} #list...items() creates a list of tuples...
        else:
            self.estimator_dict=estimator_dict
        
    def setModelDict(self):
        self.model_dict={key:val['pipe'](**val['pipe_kwargs']) for key,val in self.estimator_dict.items()}
    
        
    def runCrossValidate(self):
        
        #expand_multipipes kwarg replaced with self.run_stacked
        n_jobs=self.cv_n_jobs
        cv_results={}
        for estimator_name,model in self.model_dict.items():
            start=time()
            model_i=cross_validate(
                model, self.X_df, self.y_df, return_estimator=True, 
                scoring=self.scorer_list, cv=self.getCV(), n_jobs=n_jobs)
            end=time()
            print(f"{estimator_name},{[(scorer,np.mean(model_i[f'test_{scorer}'])) for scorer in self.scorer_list]}, runtime:{(end-start)/60} min.")
            cv_results[estimator_name]=model_i
        if self.run_stacked:
            for est_name,result in cv_results.items():
                if type(result['estimator'][0]) is MultiPipe:
                    self.logger.info(f'expanding multipipe: {est_name}')
                    new_results={}
                    for mp in result['estimator']:
                        print(mp)
                        for est_n,m in mp.build_individual_fitted_pipelines().items():
                            if not est_n in new_results:
                                new_results[est_n]=[]
                            new_results[est_n].append(m)
                    for est_n in new_results:
                        if est_n in cv_results:
                            est_n+='_fcombo'
                        self.cv_results[est_n]=new_results[est_n]
        self.cv_results=cv_results
        
    def getCV(self,cv_dict=None):
        if cv_dict is None:
            cv_dict=self.project_CV_dict
        cv_reps=cv_dict['cv_reps']
        cv_folds=cv_dict['cv_folds']
        cv_strategy=cv_dict['cv_strategy']
        if cv_strategy is None:
            return RepeatedKFold(
                n_splits=cv_folds, n_repeats=cv_reps, random_state=self.rs)
        else:
            assert type(cv_strategy) is tuple,f'expecting tuple for cv_strategy, got {cv_strategy}'
            cv_strategy,cv_groupcount=cv_strategy
            return regressor_q_stratified_cv(
                n_splits=cv_folds, n_repeats=cv_reps, 
                random_state=self.rs,groupcount=cv_groupcount,strategy=cv_strategy)

    def predictCVYhat(self,):
        train_idx_list,test_idx_list=zip(*list(self.getCV().split(self.X_df,self.y_df)))
        n,k=self.X_df.shape
        data_idx=np.arange(n)
        yhat_dict={}
        for idx,(estimator_name,result) in enumerate(self.cv_results.items()):
            yhat_dict[estimator_name]=[]
            for r in range(self.cv_reps):
                yhat=np.empty([n,])
                for s in range(self.cv_folds): # s for split
                    m=r*self.cv_folds+s
                    cv_est=result['estimator'][m]
                    test_rows=test_idx_list[m]
                    yhat[test_rows]=cv_est.predict(self.X_df.iloc[test_rows])
                yhat_dict[estimator_name].append(yhat)
        self.cv_yhat_dict=yhat_dict
        
        
    def buildCVScoreDict(self):
        try: self.cv_yhat_dict
        except:self.predictCVYhat()
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
        
        
    def plotCVYhat(self,single_plot=True):
        colors = plt.get_cmap('tab10')(np.arange(20))#['r', 'g', 'b', 'm', 'c', 'y', 'k']    
        fig=plt.figure(figsize=[12,15])
        plt.suptitle(f"Y and CV-test Yhat Across {self.cv_count} Cross Validation Runs. ")
        
        #ax.set_xlabel('estimator')
        #ax.set_ylabel(scorer)
        #ax.set_title(scorer)
        
        y=self.y_df
        n=y.shape[0]
        y_sort_idx=np.argsort(y)
        
        xidx_stack=np.concatenate([np.arange(n) for _ in range(self.cv_reps)],axis=0)
        est_count=len(self.cv_yhat_dict)
        if single_plot:
            ax=fig.add_subplot(111)
            ax.plot(np.arange(n),y.iloc[y_sort_idx],color='k',alpha=0.9,label='y')
        for e,(est_name,yhat_list) in enumerate(self.cv_yhat_dict.items()):
            if not single_plot:
                ax=fig.add_subplot(est_count,1,e+1)
                ax.plot(np.arange(n),y.iloc[y_sort_idx],color='k',alpha=0.7,label='y')
            yhat_stack=np.concatenate(yhat_list,axis=0)
            ax.scatter(xidx_stack,yhat_stack,color=colors[e],alpha=0.4,marker='_',s=7,label=f'yhat_{est_name}')
            #ax.hist(scores,density=1,color=colors[e_idx],alpha=0.5,label=estimator_name+' cv score='+str(np.mean(cv_score_dict[estimator_name][scorer])))
            ax.grid(True)
        #ax.xaxis.set_ticks([])
        #ax.xaxis.set_visible(False)
            ax.legend(loc=2)
        
        
        
        
        
        
        
        
            
    
    
        
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
            

    
        
    
