#paper: https://www.stat.cmu.edu/~ryantibs/papers/jackknife.pdf
#code: http://www.stat.uchicago.edu/~rina/jackknife/jackknife_simulation.html
#related work: https://arxiv.org/abs/2002.09025, http://www.stat.uchicago.edu/~rina/jackknife/jackknife+-after-bootstrap_realdata.html

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold

class CVPlusPI: #implements Jacknife_Plus (n_splits=train_n,cv_test_n=1) or Cross-Validation Plus (n_splits < train_n,cv_test_n=train_n/n_splits) by setting collapse_reps to 'drop'
    @staticmethod #no class attributes are used; lets you use CVPlusPI without instantiating an object of this class
    def run(y_train:np.array, #the colon allows declaration of a variable type on the fly
            cv_yhat_train:np.array,cv_yhat_predict:np.array,alpha=0.05,collapse_reps:str='pre_mean',true_y_predict=None):
        #Doug's experience leads him to believe pre-mean and drop perform the best; probably just leave this at pre-mean
        n_reps,train_n=cv_yhat_train.shape #yhat's from the observations withheld from each cv_sub-model
        assert y_train.shape[-1]==cv_yhat_train.shape[-1], 'expecting rhs dimensions of y and cv_yhat_train to match. shapes: y_train: {y_train.shape}, cv_yhat_train{cv_yhat_train.shape}'
        cv_train_err=np.abs(y_train[None,:]-cv_yhat_train) #dims of lhs is (cv_reps,train_n)
        
        '''n_splits=int(cv_yhat_predict.shape[0]/n_reps)
        assert cv_yhat_predict.shape[0]==n_reps*n_splits,f'n_splits should be an integer, but \
            n_splits:{n_splits} and cv_yhat_predict.shape:\
            {cv_yhat_predict.shape}, cv_yhat_train.shape:{cv_yhat_train.shape}'
        '''
        assert cv_yhat_predict.shape[0]==n_reps
        assert cv_yhat_predict.shape[1]==train_n
        predict_n=cv_yhat_predict.shape[-1]
        
        #cv_yhat_predict=cv_yhat_predict.reshape(n_reps,n_splits,predict_n); delete?
        assert type(collapse_reps) is str
        if collapse_reps=='drop':
            cv_yhat_predict=cv_yhat_predict[0][None,:,:] #grab 0th rep (causing a loss of 1 dim) and put back the same dim
            cv_train_err=cv_train_err[0][None,:] #same as above, but cv_train_err only has 2 dims
            n_reps=1
        elif collapse_reps=='pre_mean':
            cv_yhat_predict=np.mean(cv_yhat_predict,axis=0)[None,:,:] #take mean across reps and add lost dim back
            cv_train_err=np.mean(cv_train_err,axis=0)[None,:] #same as above, but cv_train_err only has 2 dims
            n_reps=1
        elif collapse_reps=='pre_median':
            cv_yhat_predict=np.median(cv_yhat_predict,axis=0)[None,:,:] #take median across reps and add lost dim back
            cv_train_err=np.median(cv_train_err,axis=0)[None,:] #same as above, but cv_train_err only has 2 dims
            n_reps=1
        elif collapse_reps=='post_mean':
            pass #changes happen later
        elif collapse_reps.lower()=='none':
            pass
                
        else:assert False,'collapse_reps:{collapse_reps} not developed'

        if collapse_reps=='post_mean':
            q_n=train_n*1
            alpha_idx=int(np.ceil((q_n+1)*(1-alpha)))
            lower_q=np.mean(np.sort((cv_yhat_predict-cv_train_err[:,:,None]),axis=1)[:,-alpha_idx,:],axis=0)
            #dimensions:(n_reps,train_n,predict_n)-> (n_reps,train_n,predict_n) -> (n_reps,predict_n,)-> (predict_n,)
            upper_q=np.mean(np.sort((cv_yhat_predict+cv_train_err[:,:,None]),axis=1)[:,alpha_idx-1,:],axis=0)
        else:
            q_n=train_n*n_reps
            alpha_idx=int(np.ceil((q_n+1)*(1-alpha))) #location of prediction interval quantile boundary values in the sorted vector of simulated prediction errors
            #the dims (n_reps,train_n,predict_n) re-shaped into (n_reps*train_n, predict_n) and then selected to lead to (predict_n,); the comma is necessary for a 1-item tuple
            lower_q=np.sort((cv_yhat_predict-cv_train_err[:,:,None]).reshape(n_reps*train_n,predict_n),axis=0)[-alpha_idx,:]
            upper_q=np.sort((cv_yhat_predict+cv_train_err[:,:,None]).reshape(n_reps*train_n,predict_n),axis=0)[alpha_idx-1,:]
        if not true_y_predict is None: #if you have true y's - people testing the software
            assert true_y_predict.shape[-1]==predict_n
            #coverage is the proportion of true y's found within the prediction intervals
            coverage=np.sum(np.ones_like(true_y_predict)[np.where((true_y_predict>lower_q) & (true_y_predict<upper_q))])/predict_n
    
            #print(f'predicted coverage bound 1-2*alpha:{1-2*alpha}, coverage:{100*np.round(coverage,6)}%')
            return lower_q,upper_q,coverage
        return lower_q,upper_q

    @staticmethod
    def make_data(train_n=15,k=4,predict_n=100,n_reps=10,n_splits=5): #synthetic data for testing
        n=train_n+predict_n
        beta=np.random.normal(size=k+1)*1
        X=np.concatenate([np.random.normal(size=(n,k)),np.ones((n,1))],axis=1)
        y=X@beta+np.random.normal(size=n)*1
        X_train=X[:train_n,:]
        y_train=y[:train_n]
        X_predict=X[train_n:,:]
        true_y_predict=y[train_n:]
        splitter=RepeatedKFold(n_splits=n_splits,n_repeats=n_reps)
        cv_yhat_train=np.empty((n_reps,train_n))
        #cv_yhat_predict=np.empty((n_reps*n_splits,predict_n))
        cv_yhat_predict=np.empty((n_reps,train_n,predict_n))
        r=0;s=0
        for cv_train_idx,cv_test_idx in splitter.split(X_train):
            if s==n_splits:
                s=0
                r+=1
            cv_model=LinearRegression().fit(X_train[cv_train_idx],y=y_train[cv_train_idx])
            yhat_i=cv_model.predict(X_train[cv_test_idx]) #cv_test y_hats
            cv_yhat_train[r,cv_test_idx]=yhat_i
            yhat_predict_i=cv_model.predict(X_predict) #cv_predict y_hats
            cv_yhat_predict[r,cv_test_idx,:]=yhat_predict_i
            s+=1
        return y_train,cv_yhat_train,cv_yhat_predict,true_y_predict

    @staticmethod            
    def run_comparison(collapse_options=['None','pre_mean','pre_median','post_mean','drop'],n=50,make_data_kwargs=dict(train_n=50,k=40,predict_n=1000,n_splits=5, n_reps=5),return_results=False):
        record={c_o:[] for c_o in collapse_options}
        for _ in range(n):
            print('.',sep='',end='')
            y_train,cv_yhat_train,cv_yhat_predict,true_y_predict=CVPlusPI.make_data(**make_data_kwargs)
            for c_o in collapse_options:
                record[c_o].append(CVPlusPI.run(y_train,cv_yhat_train,cv_yhat_predict,true_y_predict=true_y_predict,collapse_reps=c_o))
        print('')#for a new line
        print(*[(f'option: {c_o}','avg width: ',np.mean([l[1]-l[0] for l in r]), 'avg coverage: ',np.mean([l[2] for l in r]), 'min coverage: ',min(l[2] for l in r)) for c_o,r in record.items()],sep='\n')
        if return_results:
            return record

if __name__=="__main__":
    CVPlusPI.run_comparison(n=10)