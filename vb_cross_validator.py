import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer

class regressor_q_stratified_cv:
    def __init__(self,n_splits=10,n_repeats=2,group_count=10,random_state=0,strategy='quantile'):
        self.group_count=group_count
        self.strategy=strategy
        self.cvkwargs=dict(n_splits=n_splits,n_repeats=n_repeats,random_state=random_state)
        self.cv=RepeatedStratifiedKFold(**self.cvkwargs)
        #self.discretizer=KBinsDiscretizer(n_bins=self.group_count,encode='ordinal',strategy=self.strategy)  
            
    def split(self,X,y,groups=None):
        #kgroups=self.discretizer.fit_transform(y[:,None])[:,0]
        ysort_order=np.argsort(y)
        y1=np.ones(y.shape)
        y1split=np.array_split(y1,self.group_count)
        kgroups=np.empty(y.shape)
        kgroups[ysort_order]=np.concatenate([y1split[i]*i for i in range(self.group_count)],axis=0)
        return self.cv.split(X,kgroups)
    
    def get_n_splits(self,X,y,groups=None):
        return self.cv.get_n_splits(X,y,groups)
    
    
if __name__=="__main__":
    n_splits=5
    n_repeats=5
    group_count=5
    cv=regressor_q_stratified_cv(n_splits=n_splits,n_repeats=n_repeats,group_count=group_count,random_state=0,strategy='uniform')
    import numpy as np
    n=20
    y=np.linspace(-n//2,n//2,n+1)
    n=y.size
    np.random.shuffle(y)
    X=y.copy()[:,None] # make 2d
    
    i=0;j=0;splist=[];test_idx_list=[]
    for train,test in cv.split(X,y):
        if i==0:print(f'cv results for *test* set {j} ')
        #print(train,test)
        range_i=np.ptp(y[train])
        splist.append(range_i)
        test_idx_list.append(train)
        print(f'range for rep:{j}, fold:{i}, {range_i}')
        i+=1
        if i==n_splits:
            test_unique_count=np.size(np.unique(np.concatenate(test_idx_list)))
            print(f'range of ranges, {np.ptp(np.array(splist))}')
            print(f'unique elements:{test_unique_count} for n:{n}','\n')
            splist=[];test_idx_list=[]
            i=0;j+=1
            