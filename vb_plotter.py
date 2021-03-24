
import logging,logging.handlers

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


    def plotCVYhatVsY(self,regulatory_standard=False,decision_criteria=False):
        assert False,'not developed'
    
    
        
    def plotCVYhat(self,single_plot=True):
        cv_count=self.project_CV_dict['cv_count']
        cv_reps=self.project_CV_dict['cv_reps']
        cv_folds=self.project_CV_dict['cv_folds']
        colors = plt.get_cmap('tab10')(np.arange(20))#['r', 'g', 'b', 'm', 'c', 'y', 'k']    
        fig=plt.figure(figsize=[12,15])
        plt.suptitle(f"Y and CV test Yhat Across {cv_reps} repetitions of CV.")
        
        #ax.set_xlabel('estimator')
        #ax.set_ylabel(scorer)
        #ax.set_title(scorer)
        
        y=self.y_df
        n=y.shape[0]
        y_sort_idx=np.argsort(y)
        
        xidx_stack=np.concatenate([np.arange(n) for _ in range(cv_reps)],axis=0)
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
            #ax.hist(scores,density=1,color=colors[e_idx],alpha=0.5,label=estimator_name+' cv score='+str(np.mean(cv_score_dict[estimator_name][scorer])))
            ax.grid(True)
        #ax.xaxis.set_ticks([])
        #ax.xaxis.set_visible(False)
            ax.legend(loc=2)