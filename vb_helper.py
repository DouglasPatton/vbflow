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
from vb_pi import CVPlusPI
import json,pickle
import joblib
import sys
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
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
            datefmt='%Y-%m-%dT%H:%M:%S')
        self.logger = logging.getLogger(handlername)    
        
class VBHelper(myLogger):
    def __init__(self,drop_duplicates=False,nan_threshold=0.99,test_share=0,cv_folds=5,cv_reps=2,random_state=0,cv_strategy=None,run_stacked=True,cv_n_jobs=4,shuffle=True):       
        myLogger.__init__(self)
        self.cv_n_jobs=cv_n_jobs #the number of separate jobs to use (parallel processing) for cross-validation
        self.run_stacked=run_stacked #Boolean; indicates whether pipelines should be run as part of a stacking regressor;
        # see FCombo in vb_estimators.py
        self.setProjectCVDict(cv_folds,cv_reps,cv_strategy) #this creates a dictionary to store these values
        self.test_share=test_share #proportion of dataset set aside for final testing; these selected observations will
        # not be used in any model fitting
        self.rs=random_state #used to seed all of the random number generators wherever they occur
        self.drop_duplicates=drop_duplicates #applied to all dataset rows; can be False (do nothing), "X" (has the exact
        # same covariate values), "Xy" (same covariate values, same response); behavior varies dependent on whether unique
        # indexing for each observation is included in the uploaded data
        self.nan_threshold=nan_threshold #applied to each dataset column; can range from 0 to 1; if proportion of nan's
        # exceeds this threshold, column deleted
        self.shuffle=shuffle #Boolean; determines if rows of dataset are shuffled prior to cross-validation
                
        # below are added in the notebook
        self.scorer_list=None #list of strings that specify scikit learn scorers (model evaluation metrics where higher
        # score is better) to use for reporting model accuracy (cross-validation or testing); an example would be
        # ['neg_mean_squared_error','neg_mean_absolute_error','r2']; see https://scikit-learn.org/stable/modules/model_evaluation.html for all options
        self.max_k=None #maximun number of covariates selected by shrinking transformers like ShrinkBigTransformers in vb_transformers.py
        self.pipe_dict=None #dictionary with key:value pairs where key is the pipeline name and value contains the information to create the pipeline
        self.model_dict=None #dictionary with key:value pairs where key is the pipeline name and value contains an instantiated pipeline, aka a model
        ##
        self.predictive_models={} #stores a key:value pair where the key is the name of the model and value is the model
        
        #self.plotter=VBPlotter()
        self.jhash=None #for storing checkpoints to speed up debugging; internal use only

    def pickleSelf(self,path=None):
        if path is None:
            path='vbhelper.pkl'
        with open(path,'wb') as f:
            pickle.dump(self,f)

    def setProjectCVDict(self,cv_folds,cv_reps,cv_strategy):
        #cv_folds is the number of cross-validation folds; typical values would be 5 or 10
        #cv_reps is the number of cross-validaton repetitions; each repetition reshuffles the dataset
        #cv_strategy allows for different cross-validation strategies; options are None,('quantile',5),('uniform',5) where
        # 5 represents the group count, which depends on how much data you have; NEEDS DETAILS
        cv_count=cv_reps*cv_folds
        self.project_CV_dict={
            'cv_folds':cv_folds,
            'cv_reps':cv_reps,
            'cv_count': cv_count,
            'cv_strategy':cv_strategy
        }

    def setData(self,X_df,y_df): #X_df is the covariate data frame, y_df is the series of response data
        self.dep_var_name=y_df.columns.to_list()[0] #assigning a name to the response variable from y_df
        X_df,y_df=self.checkData(X_df,y_df) #checkData function call
        shuf=np.arange(y_df.shape[0]) #create an array (0 to the number of observations-1) to be used for shuffling the data
        if self.shuffle:
            seed=self.rs #rs is the previously stored random seed
            rng = np.random.default_rng(seed) #creating the random number generator
            rng.shuffle(shuf) #shuffling the shuf array using the random number generator
            X_df=X_df.iloc[shuf] #shuffling the X_df data
            y_df=y_df.iloc[shuf] #shuffling the y_df data

        #this condition checks if the user has set aside testing data, and if so, store it to X_test and y_test
        # and overwrites X_df and y_df to exclude those values
        if self.test_share>0:
            self.X_df,self.X_test,self.y_df,self.y_test=train_test_split(
                X_df, y_df,test_size=self.test_share,random_state=self.rs)
        else:
            self.X_df=X_df;self.y_df=y_df
            self.X_test=None;self.y_test=None

        #this condition checks for categorical variables and stores their location and name in self.cat_idx and self.cat_vars
        if 'object' in list(dict(X_df.dtypes).values()):
            self.cat_idx,self.cat_vars=zip(*[(i,var) for i,(var,dtype) in enumerate(dict(X_df.dtypes).items()) if dtype=='object'])
        else:
            self.cat_idx=[];self.cat_vars=[]

        #recording all other non-categorical variables
        self.float_idx=[i for i in range(X_df.shape[1]) if i not in self.cat_idx]

    def checkData(self,X_df,y_df):
        X_dtype_dict=dict(X_df.dtypes) #converting X_df.dtypes into a dictionary that holds the data types of the covariates
        for var,dtype in X_dtype_dict.items():
            if str(dtype)[:3]=='int':
                #print(f'changing {var} to float from {dtype}')
                X_df.loc[:,var]=X_df.loc[:,var].astype('float') #converting integer covariates to floats
        data_df=X_df.copy() #create a copy of X_df and then combine X_df.copy and y_df into a single data frame on the next line
        data_df.loc['dependent_variable',:]=y_df.loc[:,self.dep_var_name]
        X_duplicates=X_df.duplicated() #makes a new data frame where duplicates observations in X_df are assigned a boolean (True/False)
        full_duplicates=data_df.duplicated() #same as previous line, but with Xy
        full_dup_sum=full_duplicates.sum() #counts the number of duplicates in Xy (excluding the first occurrence)
        X_dup_sum=X_duplicates.sum() #same as previous line, but for X
        print(f'# of duplicate rows of data: {full_dup_sum}')
        print(f'# of duplicate rows of X: {X_dup_sum}')
        
        if self.drop_duplicates:
            if self.drop_duplicates.lower() in ['yx','xy','full']:
                X_df=X_df[~full_duplicates] #replacing X_df with a new X_df without any duplicates (first occurrence retained)
                print(f'# of duplicate Xy rows dropped: {full_dup_sum}')
            elif self.drop_duplicates.lower()=='x':
                X_df=X_df[~X_duplicates] #replacing X_df with a new X_df without any duplicates in X (first occurrence retained)
                print(f'# of duplicate X rows dropped: {X_dup_sum}')
            else:
                assert False,'unexpected drop_duplicates:{self.drop_duplicates}' #error based on unexpected value for drop_duplicates
        drop_cols=X_df.columns[X_df.isnull().sum(axis=0)/X_df.shape[0]>self.nan_threshold] #choosing the columns of X that
        # have more than a threshold proportion of missing values
        if len(drop_cols)>0:
            print(f'columns to drop: {drop_cols}')
            X_df.drop(drop_cols,axis=1,inplace=True) #deleting those columns from X_df
        else:
            print(f'no columns exceeded nan threshold of {self.nan_threshold}')

        return X_df,y_df #returns corrected data frames

    def saveFullFloatXy(self): #this function is exporting the data for initial data visualization, but this stuff won't be used for eventual pipeline training
        mvh=missingValHandler({ #create an object for cleaning the covariate data
            'impute_strategy':'impute_knn5'#'pass-through'
        })
        #the next lines do data prep, like imputation and binarizing categorical variables
        mvh=mvh.fit(self.X_df)
        X_float=mvh.transform(self.X_df)
        #create a new dataset, potentially with more columns when categorical variables were expanded
        X_float_df=pd.DataFrame(data=X_float,columns=mvh.get_feature_names(input_features=self.X_df.columns.to_list()))
        X_json_s=X_float_df.to_json() #_json_s is json-string
        y_json_s=self.y_df.to_json()
        X_nan_bool_s=self.X_df.isnull().to_json() #matrix locations in X of missing values so we can plot them
        
        summary_data={'full_float_X':X_json_s,'full_y':y_json_s, 'X_nan_bool':X_nan_bool_s} 
        self.summary_data=summary_data
        with open('summaryXy.json','w') as f:
            json.dump(summary_data,f) #saving a json of the summary data
        print(f'summary data saved to summaryXy.json')

    def setPipeDict(self,pipe_dict): #stores all the info needed to create pipelines
        self.original_pipe_dict=pipe_dict
        if self.run_stacked:
            self.pipe_dict={'stacking_reg':{'pipe':MultiPipe,'pipe_kwargs':{'pipelist':list(pipe_dict.items())}}} #list...items() creates a list of tuples
        else:
            self.pipe_dict=pipe_dict
        #creating a unique identifier for checkpoints
        self.jhash=joblib.hash([self.X_df,self.y_df,self.pipe_dict,self.project_CV_dict])    

    def setModelDict(self,pipe_dict=None): #instantiating the pipelines according to the instructions in setPipeDict
        if pipe_dict is None:
            self.model_dict={key:val['pipe'](**val['pipe_kwargs']) for key,val in self.pipe_dict.items()}
        else: 
            self.model_dict={key:val['pipe'](**val['pipe_kwargs']) for key,val in pipe_dict.items()}

    def runCrossValidate(self,try_load=True):
        if not os.path.exists('stash'):
            os.mkdir('stash')

        #expand_multipipes kwarg replaced with self.run_stacked
        n_jobs=self.cv_n_jobs
        cv_results={} #a dictionary where cross_validate output is stored
        new_cv_results={} #a dictionary where FCombo cross validation results are stored
        #pkl is a pickle file
        fname=os.path.join('stash',f'cv_{self.jhash}.pkl')
        if try_load and os.path.exists(fname):
            with open(fname,'rb') as f:
                self.cv_results=pickle.load(f)
            print('existing cv_results loaded')
            return

        for pipe_name,model in self.model_dict.items():
            start=time()
            model_i=cross_validate( #this is SciKit Learn's cv function, which clones the full model; only the sub-models are fit here
                model, self.X_df, self.y_df.iloc[:,0], return_estimator=True,
                scoring=self.scorer_list, cv=self.getCV(), n_jobs=n_jobs)
            end=time()
            #next line provides the average cross-validation scores across all sub-models
            print(f"{pipe_name},{[(scorer,np.mean(model_i[f'test_{scorer}'])) for scorer in self.scorer_list]}, runtime:{(end-start)/60} min.")
            cv_results[pipe_name]=model_i

        if self.run_stacked:
            for est_name,result in cv_results.items():
                if type(result['estimator'][0]) is MultiPipe: #checking if a sub-model is a MultiPipe
                    self.logger.info(f'expanding multipipe: {est_name}')
                    new_results={}
                    for mp in result['estimator']: #mp's are multi-pipe sub-models that are created by cross_validate()
                        #est_n and m are the two items in the tuple coming from mp.build_individual_fitted_pipelines().items()
                        for est_n,m in mp.build_individual_fitted_pipelines().items():
                            if not est_n in new_results:
                                new_results[est_n]=[]
                            new_results[est_n].append(m) #m is an FCombo
                    #this will disambiguate pipelines if they appear both inside and outside StackingRegressor()
                    for est_n in new_results:
                        if est_n in cv_results:
                            est_n+='_fcombo'
                        new_cv_results[est_n]={'estimator':new_results[est_n]}
            cv_results={**new_cv_results,**cv_results} #unpacks and then re-packs two dictionaries into a new dictionary
        self.cv_results=cv_results
        with open(fname,'wb') as f:
            pickle.dump(cv_results,f)

    def getCV(self,cv_dict=None): #returns a scikit cross-validator (a generator that yields a tuple that contains training and test indices)
        if cv_dict is None:
            cv_dict=self.project_CV_dict
        #pulling the reps, folds and strategy out of a dictionary (from the GUI)
        cv_reps=cv_dict['cv_reps']
        cv_folds=cv_dict['cv_folds']
        cv_strategy=cv_dict['cv_strategy']
        if cv_strategy is None:
            return RepeatedKFold( #this is the default cv_strategy for Web-VB
                n_splits=cv_folds, n_repeats=cv_reps, random_state=self.rs)
        else:
            assert type(cv_strategy) is tuple,f'expecting tuple for cv_strategy, got {cv_strategy}'
            cv_strategy,cv_groupcount=cv_strategy #unpacking the 2-item tuple
            return regressor_q_stratified_cv( #https://github.com/scikit-learn/scikit-learn/issues/4757#issuecomment-694924478
                #this is ensuring a spread (quantile or uniform bins) of y values amongst the various cv folds
                n_splits=cv_folds, n_repeats=cv_reps, 
                random_state=self.rs,groupcount=cv_groupcount,strategy=cv_strategy)

    #making a bunch of predictions for each observation
    def predictCVYhat(self,):
        cv_reps=self.project_CV_dict['cv_reps']
        cv_folds=self.project_CV_dict['cv_folds']
        train_idx_list,test_idx_list=zip(*list(self.getCV().split(self.X_df,self.y_df))) #unpacking train/test indices
        n,k=self.X_df.shape
        y=self.y_df.to_numpy()[:,0] #making y into a matrix with one column
        data_idx=np.arange(n) #delete this line
        yhat_dict={};err_dict={};cv_y_yhat_dict={} #initializing dictionaries
        for idx,(pipe_name,result) in enumerate(self.cv_results.items()): #remove idx and enumeration
            #adding new key value pairs (pipe_name : an empty list) to each of the following dictionaries
            yhat_dict[pipe_name]=[]
            cv_y_yhat_dict[pipe_name]=[]
            err_dict[pipe_name]=[] #y minus y-hat
            for r in range(cv_reps):
                yhat=np.empty([n,])
                err=np.empty([n,])
                for s in range(cv_folds): #s for splits/folds
                    m=r*cv_folds+s #continous counter that counts across reps and folds
                    cv_est=result['estimator'][m] #looking inside cross_validate output and getting the mth sub-model
                    test_rows=test_idx_list[m] #pulling out test data for the mth sub-model
                    yhat_arr=cv_est.predict(self.X_df.iloc[test_rows]) #making predictions (y-hats) for the mth sub-model
                    yhat[test_rows]=yhat_arr
                    err[test_rows]=y[test_rows]-yhat[test_rows] #computing residuals for the mth sub-model
                    cv_y_yhat_dict[pipe_name].append((self.y_df.iloc[test_rows,0].to_numpy(),yhat_arr))
                yhat_dict[pipe_name].append(yhat)
                err_dict[pipe_name].append(err)
                
        #yhat_dict['y']=self.y_df.to_numpy()
        self.cv_yhat_dict=yhat_dict
        #self.cv_y_yhat_dict=cv_y_yhat_dict
        self.cv_err_dict=err_dict
        
    def saveCVResults(self):
        #using arrayDictToListDict to make lists out of numpy arrays within the specified entities
        full_results=self.arrayDictToListDict(
            {
                'y':self.y_df.iloc[:,0].to_list(),
                'cv_yhat':self.cv_yhat_dict, #created by predictCVYhat
                'cv_score':self.cv_score_dict, #created by buildCVScoreDict
                'project_cv':self.project_CV_dict, #created by setProjectCVDict
                'cv_model_descrip':{} #not developed; allows addition of more data without changing the API; may remove?
            }
        )
        self.full_results=full_results
        #df=pd.DataFrame(full_results); delete?; "pd" is pandas
        path='project_cv_results.json'
        with open(path,'w') as f:
            #df.to_json(f)
            json.dump(full_results,f) #saving a json file
        print(f'cross validation results saved to {path}')
        #print('setting plotter data')
        #self.plotter.setData(full_results)

    #repacks numpy arrays into json friendly lists
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
        try: self.cv_yhat_dict #this will trigger an error if self.cv_yhat_dict not found
        except:self.predictCVYhat() #this will be triggered by any error, including typos
        cv_results=self.cv_results
        scorer_list=self.scorer_list
        cv_score_dict={}
        cv_score_dict_means={}
        y=self.y_df.iloc[:,0] #getting the y-values from a data frame (technically a data series)
        for idx,(pipe_name,result) in enumerate(cv_results.items()):
            model_idx_scoredict={}
            for scorer in scorer_list:
                scorer_kwarg=f'test_{scorer}'
                #if scorer_kwarg in result:
                #    model_idx_scoredict[scorer]=result[scorer_kwarg]
                #else:
                #creating a function (a_scorer) on the fly that uses scikit learn scorers with only y and y-hat as the inputs
                #python keyword lambda creates an anonymous inline function
                a_scorer=lambda y,yhat:get_scorer(scorer)(NullModel(),yhat,y) #get_scorer returns a scikit learn scorer that
                #takes three arguments (estimator,X,y); we are bypassing the internal prediction step (by using NullModel())
                #and only giving it y-hat and y
                score=np.array([a_scorer(y,yhat) for yhat in self.cv_yhat_dict[pipe_name]]) #score is a vector with cv_reps*cv_folds elements
                model_idx_scoredict[scorer]=score
            cv_score_dict[pipe_name]=model_idx_scoredict
            #taking the mean of the scores across all cv_iterations
            model_idx_mean_scores={scorer:np.mean(scores) for scorer,scores in model_idx_scoredict.items()}
            cv_score_dict_means[pipe_name]=model_idx_mean_scores
        #storing (cv_score_dict_means and cv_score_dict) as class attributes makes them more accessible to other parts of the vb_helper object
        self.cv_score_dict_means=cv_score_dict_means
        self.cv_score_dict=cv_score_dict

    def viewCVScoreDict(self):
        for scorer in self.scorer_list:
            print(f'scores for scorer: {scorer}:')
            for pipe_name in self.model_dict:
                print(f'    {pipe_name}:{self.cv_score_dict_means[pipe_name][scorer]}')
    
    def refitPredictiveModel(self,selected_model:str,verbose:bool=False):
        # TODO: Add different process for each possible predictive_model_type
        #self.logger = VBLogger(self.id)
        #functionality to load/save models to speed stuff up when debugging
        pjhash=joblib.hash([self.jhash,selected_model])
        path=os.path.join('stash','predictive_model-'+pjhash+'.pkl')
        if os.path.exists(path):
            try:
                with open(path,'rb') as f:
                    self.predictive_model=pickle.load(f)
                print(f'predictive models loaded from {path}')
                return
            except:
                print(f'failed to load predictive models from {path}, fitting')

        y_df=self.y_df #data frame here, but changed to 1 dimensional series below
        X_df=self.X_df
        self.logger.info("Refitting specified models for prediction...")
        try:
            predictive_model=(selected_model,self.model_dict[selected_model].fit(X_df,y_df.iloc[:,0]))
        except KeyError:
            pipe_i=self.original_pipe_dict[selected_model]
            #TODO: replace "name" with "selected_model"?
            predictive_model=(name, pipe_i['pipe'](**pipe_i['pipe_kwargs']).fit(X_df,y_df.iloc[:,0]))
        except:
            self.logger.exception('error refiting predictive model')
            assert False,f'unexpected selected_model:{selected_model}'
        #self.logger.info(f"Model:{predictive_model}")
        #predictive_model[1] = predictive_model[1].fit(X_df, y_df.iloc[:,0])
        self.predictive_model = predictive_model
        self.logger.info("Refitting model for prediction complete.")
        with open(path,'wb') as f:
            pickle.dump(self.predictive_model,f)

    #old functionality to allow weighted model averaging of fitted pipelines
    '''def setModelAveragingWeights(self):
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
            '''
    
    def getPredictionValues(self,X_df_predict):
        prediction_results=self.predict(X_df_predict)
        test_results=self.predict(self.x_test)
        collection={
            'prediction_results':results,
            'test_results':test_results,'test_y':self.y_test
        }
        return collection

    #obtaining and saving predictions from new X data
    def predictandSave(self,X_predict,scorer=None,build_PI='cvplus'):
        if scorer is None:
            scorer=self.scorer_list[0]
        #making sure predictions have informative row labels
        if not str(X_predict.index.to_list()[0])[:7].lower()=='predict':
            X_predict.index=[f'predict-{idx}' for idx in X_predict.index]
        yhat_predict=self.predict(X_predict) #predict with the fitted model created by refitPredictiveModel
        self.yhat_predict=yhat_predict
        yhat_cv_predict=self.predict(X_predict,model_type='cross_validation') #predict with cross-validation sub-models; will produce multiple yhat's
        
        if not build_PI is None:
            if build_PI.lower() in ['cv+','cvplus']:
                prediction_interval=self.doCVPlusPI(yhat_cv_predict)
            else: assert False,f'build_PI option: {build_PI} not developed'
        else: prediction_interval=None

        predictresult={
            'yhat_predict':yhat_predict.to_json(),#json strings
            'cv_yhat_predict':[yhat_predict_i.to_json() for yhat_predict_i in yhat_cv_predict],
            'X_predict':X_predict.to_json(),
            'yhat_predict_PI':{str(build_PI):prediction_interval.to_json()},#build_PI might be "cv+",None,...?
            'selected_model':self.predictive_model[0]} #the name of the predictive model
        path='project_prediction_results.json'
        with open(path,'w') as f:
            json.dump(predictresult,f)
        print(f'prediction results saved to {path}')

    #create a prediction interval for new predictions based partly on the variablity of yhat's seen for the training data through cross-validation
    def doCVPlusPI(self,yhat_cv_predict,alpha=0.05,collapse_reps='pre_mean',true_y_predict=None):
        y_train=np.array(self.full_results['y'])
        #organizing cv_yhat_predictions into a single array with an added left-hand side dimension along which the repetition varies
        cv_yhat_train_arr=np.concatenate(
            [np.array(y_list)[None,:] for y_list in self.full_results['cv_yhat'][self.predictive_model[0]]],axis=0
            ) # dims: (n_reps,train_n)
        splitter=self.getCV()
        n_splits=self.project_CV_dict['cv_folds']
        n_reps=self.project_CV_dict['cv_reps']
        yhat_cv_predict_arr=np.empty([n_reps,y_train.size,yhat_cv_predict[0].shape[-1]]) #yhat_cv_predict[0].shape[-1] is the number of predictions being made
        r=0
        s=0
        for train_idx,test_idx in splitter.split(y_train):
            assert r<n_reps
            #predictions are broadcast/repeated to each point in test_idx (about 20% of training n)
            yhat_cv_predict_arr[r,test_idx,:]=yhat_cv_predict[r*n_splits+s].to_numpy()
            s+=1
            if s==n_splits:
                r+=1
                s=0

#Ended here on 2/16
        #yhat_cv_predict_arr=np.concatenate([yhat_i[None,:] for yhat_i in yhat_cv_predict],axis=0) # None adds a new dimension 
        #     that will be used for concatenating the different predictions across reps and splits dims:(n_reps*n_splits,predict_n)
        lower,upper=CVPlusPI().run(
            y_train,cv_yhat_train_arr,yhat_cv_predict_arr,
            alpha=alpha,collapse_reps=collapse_reps,true_y_predict=true_y_predict)
        #print(lower,upper)
        df=pd.DataFrame(data={f'lowerPI-{alpha}':lower,f'upperPI-{alpha}':upper},index=yhat_cv_predict[0].index)
        #self.PIdf=df
        return df

    def predict(self, X_df_predict: pd.DataFrame,model_type:str='predictive'):
        name,est=self.predictive_model
        if model_type=='predictive':
            results=pd.Series(data=est.predict(X_df_predict),index=X_df_predict.index,name='yhat')
        elif model_type=='cross_validation':
            cv_list=[]
            for cv_i in range(self.project_CV_dict['cv_count']):
                model_cv_i=self.cv_results[name]['estimator'][cv_i]
                cv_list.append(pd.Series(data=model_cv_i.predict(X_df_predict),index=X_df_predict.index,name='yhat'))
            results=cv_list
        
        return results

    def get_test_predictions(self):
        test_results = {}
        if self.X_test is None:
            return test_results
        name, est = self.predictive_model
        r = {
            "y": self.y_test.to_list(),
            "yhat": est.predict(self.X_test)
        }
        test_results[name] = r
        return test_results
    
