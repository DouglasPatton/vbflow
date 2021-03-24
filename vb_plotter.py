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
        
        try: 
            self.project_CV_dict
            self.cv_reps=self.project_CV_dict['cv_reps']
            self.standalone=True
        except: 
            self.standalone=False
            print('VBPlotter running standalone')
            self.data_dict=None
            
    def setData(self,data_dict):
        assert not self.standalone,'setData is only for standalone operation'
        #data_dict looks like: {'cv_yhat':self.cv_yhat_dict,'cv_score':self.cv_score_dict}
        self.data_dict=data_dict
        
        cv_yhat_dict={}
        for key,val in data_dict['cv_yhat'].items():
            if type(val[0]) is list:
                cv_yhat_dict[key]=[np.array(v) for v in val]
            else:
                cv_yhat_dict[key]=np.array(val)
        self.cv_yhat_dict=cv_yhat_dict
        self.cv_score_dict={key:[np.array(v) for v in val] for key,val in data_dict['cv_score'].items()}
        yhat_stack_dict,y=self.stackCVYYhat()
        self.yhat_stack_dict=yhat_stack_dict
        self.y=y

        
        ests=list(yhat_stack_dict.keys())
        yhat_stack=yhat_stack_dict[ests[0]]
        cv_reps=yhat_stack.shape[0]/y.shape[0]
        assert not cv_reps%1,f'cv_reps should be evenly divisible by 1, but cv_reps:{cv_reps}'
        self.cv_reps=cv_reps
        
        
        


    def plotCVYhatVsY(self,regulatory_standard=False,decision_criteria=False):
        assert False,'not developed'
    
    def stackCVYYhat(self):
        cv_yhat_dict=self.cv_yhat_dict.copy()
        y=cv_yhat_dict.pop('y')
        yhat_stack_dict={}
        for e,(est_name,yhat) in enumerate(cv_yhat_dict.items()):
            y_list,yhat_list=zip(*y_yhat_tuplist)
            yhat_stack=np.concatenate(yhat_list,axis=0)
            yhat_stack_dict[est_name]=yhat_stack
        return yhat_stack_dict,y
        
    def plotCVYhat(self,single_plot=True):
        yhat_stack_dict,y=self.stackCVYYhat()
        colors = plt.get_cmap('tab10')(np.arange(20))#['r', 'g', 'b', 'm', 'c', 'y', 'k']    
        fig=plt.figure(figsize=[12,15])
        plt.suptitle(f"Y and CV-test-Yhat Across {self.cv_reps} repetitions of CV.")
        
        #ax.set_xlabel('estimator')
        #ax.set_ylabel(scorer)
        #ax.set_title(scorer)
        n=y.shape[0]
        y_sort_idx=np.argsort(y)
        
        xidx_stack=np.concatenate([np.arange(n)[y_sort_idx] for _ in range(self.cv_reps)],axis=0)# added y_sort_idx to put indices for x in same order as y will be.
        est_count=len(self.cv_yhat_dict)
        if single_plot:
            ax=fig.add_subplot(111)
            ax.plot(np.arange(n),y.iloc[y_sort_idx],color='k',alpha=0.9,label='y')
        #for e,(est_name,yhat_list) in enumerate(self.cv_yhat_dict.items()):
        for e,(est_name,y_yhat_tuplist) in enumerate(self.cv_y_yhat_dict.items()):
            y_list,yhat_list=zip(*y_yhat_tuplist) # y_list is the same y repeated
            if not single_plot:
                ax=fig.add_subplot(est_count,1,e+1)
                ax.plot(np.arange(n),y.iloc[y_sort_idx],color='k',alpha=0.7,label='y')
            yhat_stack=np.concatenate(yhat_list,axis=0)
            ax.scatter(xidx_stack,yhat_stack,color=colors[e],alpha=0.4,marker='_',s=7,label=f'yhat_{est_name}')
            ax.scatter(np.arange(n),)
            #ax.hist(scores,density=1,color=colors[e_idx],alpha=0.5,label=estimator_name+' cv score='+str(np.mean(cv_score_dict[estimator_name][scorer])))
            ax.grid(True)
        #ax.xaxis.set_ticks([])
        #ax.xaxis.set_visible(False)
            ax.legend(loc=2)