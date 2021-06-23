from time import time
import logging, logging.handlers
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json,pickle

class myLogger:
    def __init__(self,name=None):
        if name is None:
            name='vbplotter.log'
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

        
        
class VBPlotter(myLogger):
    def __init__(self):
        myLogger.__init__(self)
            
    def setData(self,data_dict):
        #data_dict looks like:  
        """{
            'y':self.y_df,
            'cv_yhat':self.cv_yhat_dict,
            'cv_score':self.cv_score_dict,
            'project_cv':self.project_CV_dict,
            'cv_model_descrip':None #not developed
        }"""
        self.data_dict=data_dict
        
            
        y=np.array(self.data_dict['y'])
        self.y=y
        self.project_CV_dict=self.data_dict['project_cv']
        self.cv_reps=self.project_CV_dict['cv_reps']
        cv_yhat_dict={}
        for key,val in data_dict['cv_yhat'].items():
            if type(val[0]) is list:
                cv_yhat_dict[key]=[np.array(v) for v in val]
            else:
                cv_yhat_dict[key]=np.array(val)
        self.cv_yhat_dict=cv_yhat_dict
        self.cv_score_dict={key:[np.array(v) for v in val] for key,val in data_dict['cv_score'].items()}
        yhat_stack_dict=self.stackCVYhat()
        self.yhat_stack_dict=yhat_stack_dict #stack reps into a single column
        self.y=y
        self.cv_score_dict=data_dict['cv_score']
        self.setScoreDict()
        
    def setPredictData(self,predictresults):
        self.predict_results=predictresults
        self.ypredict=pd.read_json(predictresults['yhat'])
        self.cv_ypredict=[pd.read_json(cv_i) for cv_i in predictresults['cv_yhat']]
        self.selected_estimators=predictresults['selected_models']
        
        
        


    def plotCVYhatVsY(self,single_plot=True,include_all_cv=True,regulatory_standard=False,decision_criteria=False,ypredict=False,cv_ypredict=False,estimators='all',true_y=None):
        #true y on horizontal axis, yhat on vertical axis
        yhat_stack_dict=self.yhat_stack_dict
        y=self.y
        
        colors = plt.get_cmap('tab10')(np.arange(10))#['r', 'g', 'b', 'm', 'c', 'y', 'k']    
        fig=plt.figure(figsize=[12,8],dpi=200)
        plt.suptitle(f"CV-test-Yhat Vs. Y Across {self.cv_reps} repetitions of CV.")
        
        #ax.set_xlabel('estimator')
        #ax.set_ylabel(scorer)
        #ax.set_title(scorer)
        n=y.shape[0]
        #ymin=y.min()
        #ymax=y.max()
        #y_sort_idx=np.argsort(y) #other orderings could be added
        #y_sort_idx_stack=np.concatenate([y_sort_idx for _ in range(self.cv_reps)],axis=0)
        y_stack=np.concatenate([y for _ in range(self.cv_reps)],axis=0)
        #xidx_stack=np.concatenate([np.arange(n)[y_sort_idx] for _ in range(self.cv_reps)],axis=0)# added y_sort_idx to put indices for x in same order as y will be.
        est_count=len(self.cv_yhat_dict)
        if single_plot:
            ax=fig.add_subplot(111)
            ax.scatter(y,y,s=20,alpha=0.4,label='y',zorder=0,color='k')
            ax.set_xlabel('observed Y')
            ax.set_ylabel('predicted Y')
        #for e,(est_name,yhat_list) in enumerate(self.cv_yhat_dict.items()):
        for e,(est_name,yhat_stack) in enumerate(self.yhat_stack_dict.items()):
            if not estimators=='all':
                if estimators=='selected':
                    if not est_name in self.selected_estimators:continue
                else:assert False,'not developed'
            if ypredict or cv_ypredict:
                all_y=np.concatenate([y,yhat_stack],axis=0)
                ymin=all_y.min()
                ymax=all_y.max()
            if not single_plot:
                ax=fig.add_subplot(est_count,1,e+1)
                ax.scatter(y,y,s=20,alpha=0.6,label='y',zorder=0,color='k')
                #ax.scatter(np.arange(n),y[y_sort_idx],color='k',alpha=0.7,size=1.5,label='y')
            #yhat_stack=np.concatenate(yhat_list,axis=0)
            if include_all_cv:
                ax.scatter(y_stack,yhat_stack,color=colors[e],alpha=0.2,s=2,marker='_',label=f'cv_yhat_{est_name}',zorder=1)
            
            ax.scatter(
                y,yhat_stack.reshape(self.cv_reps,n).mean(axis=0),
                s=20,marker='*',color=colors[e],alpha=0.7,label=f'mean_cv_yhat_{est_name}',zorder=2)
            #ax.hist(scores,density=1,color=colors[e_idx],alpha=0.5,label=estimator_name+' cv score='+str(np.mean(cv_score_dict[estimator_name][scorer])))
            
            if ypredict:
                
                yhat_df=self.ypredict
                for i,idx in enumerate(yhat_df.index):
                    ax.hlines(yhat_df.loc[idx],ymin,ymax,label=idx,color=colors[i])
            if cv_ypredict:
                cv_yhat_list=self.cv_ypredict
                for cv_i,cv_yhat_df in enumerate(cv_yhat_list):
                    for i,idx in enumerate(cv_yhat_df.index):
                        label_i = f'cv_{idx}' if cv_i==0 else None
                        ax.hlines(
                            cv_yhat_df.loc[idx],ymin,ymax,
                            color=colors[i],alpha=0.15,linestyles='--',
                            label=label_i)
                    
            if not true_y is None:
                for i,y in enumerate(true_y):
                    ax.vlines(
                        y,ymin,ymax,
                        label=f'true_y-{true_y.index[i]}',
                        color=colors[i],linestyles='---')
                    
            ax.grid(True)
        #ax.xaxis.set_ticks([])
        #ax.xaxis.set_visible(False)
            ax.legend(loc=2)
            
        fig.tight_layout()
        fig.show()
    
    def stackCVYhat(self):
        #makes a single column of all yhats across cv iterations for graphing
        #returns a dictionary with model/estimator/pipeline name as the key

        y=self.y
        yhat_stack_dict={}
        for e,(est_name,yhat_list) in enumerate(self.cv_yhat_dict.items()):
            yhat_stack=np.concatenate(yhat_list,axis=0)
            yhat_stack_dict[est_name]=yhat_stack
        return yhat_stack_dict
    
    def setScoreDict(self):
        scorer_score_dict={}

        for pipe_name,score_dict in self.cv_score_dict.items():
            for scorer,score_arr in score_dict.items():
                if not scorer in scorer_score_dict:
                    scorer_score_dict[scorer]={}
                scorer_score_dict[scorer][pipe_name]=score_arr
      
        self.score_dict=scorer_score_dict
        
    def plotCVYhat(self,single_plot=True,include_all_cv=True):
        yhat_stack_dict=self.yhat_stack_dict
        y=self.y
        colors = plt.get_cmap('tab10')(np.arange(10))#['r', 'g', 'b', 'm', 'c', 'y', 'k']    
        fig=plt.figure(figsize=[12,8],dpi=200)
        plt.suptitle(f"Y and CV-test-Yhat Across {self.cv_reps} repetitions of CV.")
        
        #ax.set_xlabel('estimator')
        #ax.set_ylabel(scorer)
        #ax.set_title(scorer)
        n=y.shape[0]
        y_sort_idx=np.argsort(y) #other orderings could be added
        y_sort_idx_stack=np.concatenate([y_sort_idx for _ in range(self.cv_reps)],axis=0)
        xidx_stack=np.concatenate([np.arange(n) for _ in range(self.cv_reps)],axis=0)
        #xidx_stack=np.concatenate([np.arange(n)[y_sort_idx] for _ in range(self.cv_reps)],axis=0)# added y_sort_idx to put indices for x in same order as y will be.
        est_count=len(self.cv_yhat_dict)
        if single_plot:
            ax=fig.add_subplot(111)
            ax.plot(y[y_sort_idx],'.-k',alpha=0.6,label='y',zorder=0)
        #for e,(est_name,yhat_list) in enumerate(self.cv_yhat_dict.items()):
        for e,(est_name,yhat_stack) in enumerate(self.yhat_stack_dict.items()):
            if not single_plot:
                ax=fig.add_subplot(est_count,1,e+1)
                ax.plot(y[y_sort_idx],'.-k',alpha=0.6,label='y',zorder=0)
                #ax.scatter(np.arange(n),y[y_sort_idx],color='k',alpha=0.7,size=1.5,label='y')
            #yhat_stack=np.concatenate(yhat_list,axis=0)
            if include_all_cv:
                ax.scatter(xidx_stack,yhat_stack[y_sort_idx_stack],color=colors[e],alpha=0.2,s=2,marker='_',label=f'cv_yhat_{est_name}',zorder=1)
            
            ax.scatter(
                np.arange(n),yhat_stack.reshape(self.cv_reps,n).mean(axis=0)[y_sort_idx],
                s=20,marker='*',color=colors[e],alpha=0.7,label=f'mean_cv_yhat_{est_name}',zorder=2)
            #ax.hist(scores,density=1,color=colors[e_idx],alpha=0.5,label=estimator_name+' cv score='+str(np.mean(cv_score_dict[estimator_name][scorer])))
            ax.grid(True)
        #ax.xaxis.set_ticks([])
        #ax.xaxis.set_visible(False)
            ax.legend(loc=2)
            
        fig.tight_layout()
        fig.show()
        
        
    def printCVScoreDict(self):
        for pipe_name,score_dict in self.cv_score_dict.items():
            pass
        for scorer in self.scorer_list:
            print(f'scores for scorer: {scorer}:')
            for pipe_name in self.model_dict:
                print(f'    {pipe_name}:{self.cv_score_dict_means[pipe_name][scorer]}')
    
    def plotBoxWhiskerCVScores(self,):
        
        fig=plt.figure(figsize=[12,8])
        plt.suptitle('Box Whisker plots for fit of each pipeline',fontsize=14)
        for s_idx,(scorer,pipe_scores) in enumerate(self.score_dict.items()):
            ax=fig.add_subplot(len(self.score_dict),1,s_idx+1)
            df=pd.DataFrame(pipe_scores)
            df.boxplot(ax=ax)
            ax.title.set_text(scorer)
        fig.tight_layout()
        fig.show()
        

    
    def plotCVScores(self,sort=1):
        colors = plt.get_cmap('tab10')(np.arange(10))#['r', 'g', 'b', 'm', 'c', 'y', 'k']    
        fig=plt.figure(figsize=[12,8])
        plt.suptitle(f"Model Scores Across {self.cv_reps} Cross Validation repetitions. ")
        s_count=len(self.score_dict)
        for s_idx,(scorer,pipe_scores) in enumerate(self.score_dict.items()):
            ax=fig.add_subplot(s_count,1,s_idx+1)
            ax.set_title(scorer)
            df=pd.DataFrame(pipe_scores)
            if sort:
                for col in df:
                    df[col]=df[col].sort_values(ignore_index=True)
            df.plot(ax=ax)
        fig.tight_layout()
        fig.show() 
        
        