from time import time
import logging, logging.handlers
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, LinearRegression, Lars
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, make_scorer,get_scorer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_validate, train_test_split, RepeatedKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer,KNNImputer
from vb_estimators import MultiPipe,FCombo,NullModel
from missing_val_transformer import missingValHandler
from vb_plotter import VBPlotter
import json,pickle
import joblib

from scipy.stats import spearmanr
from scipy.cluster import hierarchy

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
        
        
class VBHelper(myLogger):
    def __init__(self,test_share=0.2,cv_folds=5,cv_reps=2,random_state=0,cv_strategy=None,run_stacked=True,cv_n_jobs=8):
        
        myLogger.__init__(self)
        self.cv_n_jobs=cv_n_jobs
        self.run_stacked=run_stacked
        self.setProjectCVDict(cv_folds,cv_reps,cv_strategy)
        self.test_share=test_share
        self.rs=random_state
        #below attributes moved to self.project_cv_dict
        #self.cv_folds=cv_folds
        #self.cv_reps=cv_reps
        #self.project_CV_dict['cv_count']=cv_reps*cv_folds
        #self.cv_strategy=cv_strategy
        #self.cv_groupcount=cv_groupcount
        
        
        # below are added in the notebook
        self.scorer_list=None
        self.max_k=None
        self.pipe_dict=None
        self.model_dict=None
        ##
        self.prediction_model_type= "average"
        self.model_averaging_weights=None
        #self.logger=logging.getLogger()
        #super().__init__() #instantiates parents (e.g., VBPlotter)
        self.plotter=VBPlotter()
    
    
    def pickleSelf(self,path=None):
        if path is None:
            path='vbhelper.pkl'
        with open(path,'wb') as f:
            pickle.dump(self,f)
    
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
        if self.test_share>0:
            self.X_df,self.X_test,self.y_df,self.y_test=train_test_split(
                X_df, y_df,test_size=self.test_share,random_state=self.rs)
        else:
            self.X_df=X_df;self.y_df=y_df
            self.X_test=None;self.y_test=None
        self.cat_idx,self.cat_vars=zip(*[(i,var) for i,(var,dtype) in enumerate(dict(X_df.dtypes).items()) if dtype=='object'])
        self.float_idx=[i for i in range(X_df.shape[1]) if i not in self.cat_idx]

        
    def summarize(self):
        self.hierarchicalDendogram()
    
    
    def hierarchicalDendogram(self):
        #from https://scikit-learn.org/dev/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py
        try: self.X_float_df
        except:self.floatifyX()
        X=self.X_float_df#.to_numpy()
        plt.rcParams['font.size'] = '8'
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8),dpi=200)
        corr = spearmanr(X,nan_policy='omit').correlation
        corr_linkage = hierarchy.ward(corr)
        dendro = hierarchy.dendrogram(
            corr_linkage, labels=X.columns.tolist(), ax=ax1, leaf_rotation=90
        )
        dendro_idx = np.arange(0, len(dendro['ivl']))

        ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
        ax2.set_xticks(dendro_idx)
        ax2.set_yticks(dendro_idx)
        ax2.set_xticklabels(dendro['ivl'], rotation='vertical',fontsize=6)
        ax2.set_yticklabels(dendro['ivl'],fontsize=6)
        fig.tight_layout()
        plt.show()
        
    def floatifyX(self):
        mvh=missingValHandler({
            'impute_strategy':'impute_knn5'#'pass-through'
        })
        mvh=mvh.fit(self.X_df)
        X_float=mvh.transform(self.X_df)
        X_float_df=pd.DataFrame(data=X_float,columns=mvh.get_feature_names(input_features=self.X_df.columns.to_list()))
        self.X_float_df=X_float_df
        self.mvh=mvh
        #return X_float_df
        
    
    def setPipeDict(self,pipe_dict):
        if self.run_stacked:
            self.pipe_dict={'multi_pipe':{'pipe':MultiPipe,'pipe_kwargs':{'pipelist':list(pipe_dict.items())}}} #list...items() creates a list of tuples...
        else:
            self.pipe_dict=pipe_dict
        
    def setModelDict(self,pipe_dict=None):
        if pipe_dict is None:
            self.model_dict={key:val['pipe'](**val['pipe_kwargs']) for key,val in self.pipe_dict.items()}
        else: 
            self.model_dict={key:val['pipe'](**val['pipe_kwargs']) for key,val in pipe_dict.items()}
            
    """def implement_pipe(self,pipe_dict=None):
        else:
            return pipe_dict['pipe]']"""
    
    '''def fitFinalModelDict(self,):
        for pipe_name,pipe in self.model_dict.items():
            pipe.fit(self.X_df,self.y_df)'''
        
    def runCrossValidate(self,try_load=True):
        if not os.path.exists('stash'):
            os.mkdir('stash')
        
        
        #expand_multipipes kwarg replaced with self.run_stacked
        n_jobs=self.cv_n_jobs
        cv_results={};new_cv_results={}
        jhash=joblib.hash([self.X_df,self.y_df,self.pipe_dict,self.project_CV_dict]) #unique identifier
        if try_load:
            print("jhash: ",jhash)
        #jhash2=joblib.hash([self.X_df,self.y_df,self.pipe_dict]) 
        #print('jhash',jhash)
        #print('jhash2',jhash2)
        fname=os.path.join('stash',f'cv_{jhash}.pkl')
        if try_load and os.path.exists(fname):
            with open(fname,'rb') as f:
                self.cv_results=pickle.load(f)
            return
        
        for pipe_name,model in self.model_dict.items():
            start=time()
            model_i=cross_validate(
                model, self.X_df, self.y_df, return_estimator=True, 
                scoring=self.scorer_list, cv=self.getCV(), n_jobs=n_jobs)
            end=time()
            print(f"{pipe_name},{[(scorer,np.mean(model_i[f'test_{scorer}'])) for scorer in self.scorer_list]}, runtime:{(end-start)/60} min.")
            cv_results[pipe_name]=model_i
        if self.run_stacked:
            for est_name,result in cv_results.items():
                if type(result['estimator'][0]) is MultiPipe:
                    self.logger.info(f'expanding multipipe: {est_name}')
                    new_results={}
                    for mp in result['estimator']:
                        for est_n,m in mp.build_individual_fitted_pipelines().items():
                            if not est_n in new_results:
                                new_results[est_n]=[]
                            new_results[est_n].append(m)
                            lil_x=self.X_df.iloc[0:2]
                            print(f'est_n yhat test: {m.predict(lil_x)}')
                    for est_n in new_results:
                        if est_n in cv_results:
                            est_n+='_fcombo'
                        new_cv_results[est_n]={'estimator':new_results[est_n]}
            cv_results={**new_cv_results,**cv_results}
        self.cv_results=cv_results
        with open(fname,'wb') as f:
            pickle.dump(cv_results,f)
        
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
        cv_reps=self.project_CV_dict['cv_reps']
        cv_folds=self.project_CV_dict['cv_folds']
        train_idx_list,test_idx_list=zip(*list(self.getCV().split(self.X_df,self.y_df)))
        n,k=self.X_df.shape
        y=self.y_df.to_numpy()
        data_idx=np.arange(n)
        yhat_dict={};err_dict={};cv_y_yhat_dict={}
        for idx,(pipe_name,result) in enumerate(self.cv_results.items()):
            yhat_dict[pipe_name]=[]
            cv_y_yhat_dict[pipe_name]=[]
            err_dict[pipe_name]=[]
            for r in range(cv_reps):
                yhat=np.empty([n,])
                err=np.empty([n,])
                for s in range(cv_folds): # s for split
                    m=r*cv_folds+s
                    cv_est=result['estimator'][m]
                    test_rows=test_idx_list[m]
                    yhat_arr=cv_est.predict(self.X_df.iloc[test_rows])
                    yhat[test_rows]=yhat_arr
                    err[test_rows]=y[test_rows]-yhat[test_rows]
                    cv_y_yhat_dict[pipe_name].append((self.y_df.iloc[test_rows].to_numpy(),yhat_arr))
                yhat_dict[pipe_name].append(yhat)
                err_dict[pipe_name].append(err)
                
        #yhat_dict['y']=self.y_df.to_numpy()
            
        self.cv_yhat_dict=yhat_dict
        #self.cv_y_yhat_dict=cv_y_yhat_dict
        self.cv_err_dict=err_dict
        
    def jsonifyProjectCVResults(self):
        full_results=self.arrayDictToListDict(
            {
                'y':self.y_df.to_list(),
                'cv_yhat':self.cv_yhat_dict,
                'cv_score':self.cv_score_dict,
                'project_cv':self.project_CV_dict,
                'cv_model_descrip':{} #not developed
            }
        )
        self.full_results=full_results
        #df=pd.DataFrame(full_results)
        with open('project_cv_results.json','w') as f:
            #df.to_json(f)
            json.dump(full_results,f)
        print('setting plotter data')
        self.plotter.setData(full_results)
            
    
    def arrayDictToListDict(self,arr_dict):
        assert type(arr_dict) is dict,f'expecting dict but type(arr_dict):{type(arr_dict)}'
        list_dict={}
        for key,val in arr_dict.items():
            if type(val) is dict:
                list_dict[key]=self.arrayDictToListDict(val)
            elif type(val) is np.ndarray:
                list_dict[key]=val.tolist()
            elif type(val) is list and type(val[0]) is np.ndarray:
                list_dict[key]=[v.tolist() for v in val]
            else:
                list_dict[key]=val
        return list_dict
        
        
    def buildCVScoreDict(self):
        try: self.cv_yhat_dict
        except:self.predictCVYhat()
        cv_results=self.cv_results
        scorer_list=self.scorer_list
        cv_score_dict={}
        cv_score_dict_means={}
        y=self.y_df
        for idx,(pipe_name,result) in enumerate(cv_results.items()):
            #cv_estimators=result['estimator']
            model_idx_scoredict={}
            for scorer in scorer_list:
                scorer_kwarg=f'test_{scorer}'
                #if scorer_kwarg in result:
                #    model_idx_scoredict[scorer]=result[scorer_kwarg]
                #else:
                a_scorer=lambda y,yhat:get_scorer(scorer)(NullModel(),yhat,y) #b/c get_scorer wants (est,x,y)
                score=np.array([a_scorer(y,yhat) for yhat in self.cv_yhat_dict[pipe_name]])#
                model_idx_scoredict[scorer]=score
            #nmodel_idx_scoredict={scorer:result[f'test_{scorer}'] for scorer in scorer_list}# fstring bc how cross_validate stores list of metrics
            cv_score_dict[pipe_name]=model_idx_scoredict 
            model_idx_mean_scores={scorer:np.mean(scores) for scorer,scores in model_idx_scoredict.items()}
            cv_score_dict_means[pipe_name]=model_idx_mean_scores
        self.cv_score_dict_means=cv_score_dict_means
        self.cv_score_dict=cv_score_dict
    
   
        
    def viewCVScoreDict(self):
        for scorer in self.scorer_list:
            print(f'scores for scorer: {scorer}:')
            for pipe_name in self.model_dict:
                print(f'    {pipe_name}:{self.cv_score_dict_means[pipe_name][scorer]}')
    
    def refitPredictiveModels(self, selected_models: list,  verbose: bool=False):
        # TODO: Add different process for each possible predictive_model_type
        #self.logger = VBLogger(self.id)
        y_df=self.y_df
        X_df=self.X_df
        self.logger.info("Refitting specified models for prediction...")
        predictive_models = {}
        for name in selected_models:
            self.logger.info(f"Name: {name}")
            predictive_models[name] = self.model_dict[name]#copy.copy(self.cv_results[name]["estimator"][indx])
        self.logger.info(f"Models:{predictive_models}")
        for name, est in predictive_models.items():
            predictive_models[name] = est.fit(X_df, y_df)
        self.predictive_models = predictive_models
        self.logger.info("Refitting model for prediction complete.")

    def setModelAveragingWeights(self):
        pipe_names=list(self.predictive_models.keys())
        model_count=len(self.predictive_models)
        if self.prediction_model_type == "average":
            self.model_averaging_weights={
                pipe_names[i]:{
                    scorer:1/model_count for scorer in self.scorer_list
                } for i in range(model_count)
            }
            return
        elif self.prediction_model_type == "cv-weighted":
            totals = {
                "neg_mean_squared_error": 0,
                "neg_mean_absolute_error": 0,
                "r2": 0
            }
            value = {
                "neg_mean_squared_error": 0,
                "neg_mean_absolute_error": 0,
                "r2": 0
            }
            for name, p in self.cv_score_dict_means.items():
                if name in pipe_names: #leave out non-selected pipelines
                    totals["neg_mean_squared_error"] += 1/abs(p["neg_mean_squared_error"])
                    totals["neg_mean_absolute_error"] += 1/abs(p["neg_mean_absolute_error"])
                    totals["r2"] += p["r2"] if p["r2"] > 0 else 0
            weights = {}
            for pipe_name in pipe_names:
                weights[pipe_name] = {}
                for scorer, score in self.cv_score_dict_means[pipe_name].items():
                    # logger.warning(f"Scorer: {scorer}, Score: {score}")
                    if "neg" == scorer[:3]:
                        w = (1/(abs(score)))/totals[scorer]
                    elif scorer == "r2":
                        score = score if score > 0 else 0
                        w = score / totals[scorer]
                    else:
                        w = abs(score) / totals[scorer]

                    weights[pipe_name][scorer] = w
            self.model_averaging_weights=weights
            return
        else:
            assert False,'option not recognized'
    
    def getPredictionValues(self,X_df_predict):
        prediction_results=self.predict(X_df_predict)
        test_results=self.predict(self.x_test)
        collection={
            'prediction_results':results,
            'test_results':test_results,'test_y':self.y_test
        }
        return collection
        
    
    def predict(self, X_df_predict: pd.DataFrame,model_type:str='predictive'):
        if self.model_averaging_weights is None:
            self.setModelAveragingWeights()
        results = {}
        
        if model_type=='cross_validation':
            wtd_yhats=[{scorer:np.zeros(X_df_predict.shape[0]) for scorer in self.scorer_list} for _ in range(self.project_CV_dict['cv_count'])]
        else:
            wtd_yhats={scorer:np.zeros(X_df_predict.shape[0]) for scorer in self.scorer_list}
        for name, est in self.predictive_models.items():
            if model_type=='predictive':
                results[name] = est.predict(X_df_predict)
                for scorer,weights in self.model_averaging_weights[name].items():
                    wtd_yhats[scorer] += weights * results[name]
            elif model_type=='cross_validation':
                results[name]=[]
                for cv_i in range(self.project_CV_dict['cv_count']):
                    model_cv_i=self.cv_results[name]['estimator'][cv_i]
                    results[name].append(model_cv_i.predict(X_df_predict))
                    for scorer,weights in self.model_averaging_weights[name].items():
                        wtd_yhats[cv_i][scorer] += weights * results[name][cv_i]
        results["weights"] = self.model_averaging_weights
        results["prediction"] = wtd_yhats
        #results["final-test-predictions"] = self.get_test_predictions()
        return results

    def get_test_predictions(self):
        test_results = {}
        if self.X_test is None:
            return test_results
        for name, est in self.predictive_models.items():
            r = {
                "y": self.y_test.to_list(),
                "yhat": est.predict(self.X_test)
            }
            test_results[name] = r
        return test_results
    
