from mylogger import myLogger
class regressor_stratified_cv(myLogger)
    def __init__(self,cv_folds=10,cv_reps=2,,strategy='balanced',group_count=5,random_state=0,shuffle=True):
        #self.cv_folds=cv_folds
        #self.cv_reps=cv_reps
        self.cat_reg=cat_reg
        self.strategy=strategy
        self.group_count=group_count
        #self.random_state=random_state
        #self.shuffle=shuffle
        cvkwargs=dict(cv_folds=cv_folds,cv_reps=cv_reps,shuffle=shuffle,random_state=random_state)
        if strategy=='balanced':
            self.cv=self.RepeatStratifiedKFold(**cvkwargs)
        else:
            self.cv=self.RepeatKFold(**cvkwargs)
            
    def split(X,y,groups=None):
        if self.strategy='balanced':
            ysort_idx=np.argsort(y)
            split=np.array_split(np.ones(y.shape))
            groupsplit=[i*split[i] for i in range(self.group_count)]
            groups=np.concatenate(groupsplit,axis=0)
        return self.cv.split(X,y,groups)
                
                    
    """
        
    def old(self,):
        # test data already removed
        n,k=X_train.shape
        onecount=int(ydataarray.sum())
        sum_count_arr=np.ones([n,])
        if cat_reg=='cat':
            cats=np.unique(y_train)
            cat_count_dict={}
            for cat in cats:
                cat_count_dict[cat]=np.sum(sum_count_arr[y_train==cat])
        elif cat_reg=='reg':
            
            cat_edges=np.quantile(y_train,np.linspace(0,1,6))
            q=np.quantile(y_train)    
            #bins,_=np.histogram(y_train,bins=q)
        zerocount=n-onecount
        countlist=[zerocount,onecount]
        if onecount<zerocount:
            smaller=1
        else:
            smaller=0
        
        if not min_y is None:
            if min_y<1:
                min_y=int(batch_n*min_y)
            batch01_n=[None,None]
            batch01_n[smaller]=min_y # this makes it max_y too....
            batch01_n[1-smaller]=batch_n-min_y
            max_batchcount=countlist[smaller]//min_y
            if max_batchcount<batchcount:
                #self.logger.info(f'for {self.species} batchcount changed from {batchcount} to {max_batchcount}')
                batchcount=max_batchcount
            subsample_n=batchcount*batch_n
            oneidx=np.arange(n)[ydataarray==1]
            zeroidx=np.arange(n)[ydataarray==0]
            bb_select_list=[]
            for bb_idx in range(cv_reps):
                ones=np.random.choice(oneidx,size=batch01_n[1]*batchcount,replace=False)
                zeros=np.random.choice(zeroidx,size=batch01_n[0]*batchcount,replace=False)
                bb_select_list.append(np.concatenate([ones,zeros],axis=0))
        else:
            max_batchcount=n//batch_n
            if max_batchcount<batchcount:
                #self.logger.info(f'for {self.species} batchcount changed from {batchcount} to {max_batchcount}')
                batchcount=max_batchcount
            subsample_n=batchcount*batch_n
            for bb_idx in range(cv_reps):
                bb_select_list.append(np.random.choice(np.arange(n),size=subsample_n),replace=False)
        batchbatchlist=[[None for __ in range(batchcount)] for _ in range(cv_reps)]
        SKF=StratifiedKFold(n_splits=batchcount, shuffle=False)
        for bb_idx in range(cv_reps):
            bb_x_subsample=xdataarray[bb_select_list[bb_idx],:]
            bb_y_subsample=ydataarray[bb_select_list[bb_idx]]
            for j,(train_index,test_index) in enumerate(SKF.split(bb_x_subsample,bb_y_subsample)):
                batchbatchlist[bb_idx][j]=(bb_y_subsample[test_index],bb_x_subsample[test_index,:])
        batchsize=batch_n*batchcount
        
        
        self.batchcount=batchcount
        self.expand_datagen_dict('batchcount',self.batchcount)
        fullbatchbatch_n=cv_reps*batchsize
        self.fullbatchbatch_n=fullbatchbatch_n
        self.expand_datagen_dict('fullbatchbatch_n',self.fullbatchbatch_n)
        self.logger.info(f'yxtup shapes:{[(yxtup[0].shape,yxtup[1].shape) for yxtuplist in batchbatchlist for yxtup in yxtuplist]}')
        self.yxtup_batchbatch=batchbatchlist
        
        """