import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

class regressor_q_stratified_cv:
    def __init__(self,n_splits=10,n_repeats=2,group_count=5,random_state=0):
        self.group_count=group_count
        cvkwargs=dict(n_splits=n_splits,n_repeats=n_repeats,random_state=random_state)
        self.cv=RepeatedStratifiedKFold(**cvkwargs)
            
    def split(self,X,y,groups=None):
        split1=np.array_split(np.ones(y.shape),self.group_count)
        groupsplit=[i*split1[i] for i in range(self.group_count)]
        unsort_to_y=np.argsort(np.argsort(y)) # b/c groups is created as if y is sorted, this applies the
        #    "unsort ordering of y" to groups so groups matches the unsorted order of y
        #    i.e., if y_sorted==y[np.argsort(y)], then  y_sorted[np.argsort(np.argsort(y))]==y
        qgroups=np.concatenate(groupsplit,axis=0)[unsort_to_y]
        return self.cv.split(X,qgroups,groups)
    
    def get_n_splits(self,X,y,groups=None):
        return self.cv.get_n_splits(X,y,groups)
    
    
if __name__=="__main__":
    n_splits=5
    n_repeats=5
    group_count=5
    cv=regressor_q_stratified_cv(n_splits=n_splits,n_repeats=n_repeats,group_count=group_count,random_state=0)
    import numpy as np
    n=1000000
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
            