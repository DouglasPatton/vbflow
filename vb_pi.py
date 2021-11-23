# from http://www.stat.uchicago.edu/~rina/jackknife/jackknife_simulation.html
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold
class CVPlusPI:
    def __init__(self,):
        pass
    
    @staticmethod
    def run(y_train,cv_yhat_train,cv_yhat_predict,alpha=0.05,collapse_reps=False,true_y_predict=None):
        n_reps,train_n=cv_yhat_train.shape
        assert y_train.shape[-1]==cv_yhat_train.shape[-1],\
        f'expecting rhs dimensions of y and cv_yhat_train to match. shapes: \
            y_train: {y_train.shape}, cv_yhat_train{cv_yhat_train.shape}'
        cv_train_err=np.abs(y_train[None,:]-cv_yhat_train)# (cn_reps,train_n)
        
        n_splits=int(cv_yhat_predict.shape[0]/n_reps)
        assert cv_yhat_predict.shape[0]==n_reps*n_splits,f'n_splits should be an integer, but \
            n_splits:{n_splits} and cv_yhat_predict.shape:\
            {cv_yhat_predict.shape}, cv_yhat_train.shape:{cv_yhat_train.shape}'
        
        predict_n=cv_yhat_predict.shape[-1]
        cv_yhat_predict=cv_yhat_predict.reshape(n_reps,n_splits,predict_n)
        if collapse_reps:
            cv_yhat_predict=np.mean(cv_yhat_predict,axis=0)[None,:,:] #across reps and add dim back
            cv_train_err=np.mean(cv_train_err,axis=0)[None,:] # across reps and add dim back
            n_reps=1
        q_n=train_n*n_reps*n_splits
        alpha_idx=int(np.ceil((q_n+1)*(1-alpha)))
        lower_q=np.sort((cv_yhat_predict[:,:,None,:]-cv_train_err[:,None,:,None]).reshape(n_reps*n_splits*train_n,predict_n),axis=0)[-alpha_idx,:]
        upper_q=np.sort((cv_yhat_predict[:,:,None,:]+cv_train_err[:,None,:,None]).reshape(n_reps*n_splits*train_n,predict_n),axis=0)[alpha_idx-1,:]
        if not true_y_predict is None:
            assert true_y_predict.shape[-1]==predict_n
            coverage=np.sum(np.ones_like(true_y_predict)[np.where((true_y_predict>lower_q) & (true_y_predict<upper_q))])/predict_n
    
            print(f'predicted coverage bound 1-2*alpha:{1-2*alpha}, coverage:{np.round(coverage,2)}')
        
        
if __name__=="__main__":
    train_n=50
    k=40
    predict_n=200
    n=train_n+predict_n
    n_splits=10
    n_reps=3
    beta=np.random.normal(size=k+1)*3
    X=np.concatenate([np.random.normal(size=(n,k)),np.ones((n,1))],axis=1)
    y=X**2@beta+np.random.normal(size=n)
    X_train=X[:train_n,:]
    #print(X_train.shape)
    y_train=y[:train_n]
    X_predict=X[train_n:]
    true_y_predict=y[train_n:]
    splitter=RepeatedKFold(n_splits=n_splits,n_repeats=n_reps)
    #splitter.split(X_train)
    cv_yhat_train=np.empty((n_reps,train_n))
    cv_yhat_predict=np.empty((n_reps*n_splits,predict_n))
    r=0;s=0
    for cv_train_idx,cv_test_idx in splitter.split(X_train):
        
        if s==n_splits:
            s=0
            r+=1
            #split=splitter.split(X_train)
            #print(split)
        #print(cv_train_idx)
        cv_model=LinearRegression().fit(X_train[cv_train_idx],y=y_train[cv_train_idx])
        yhat_i=cv_model.predict(X_train[cv_test_idx])
        cv_yhat_train[r,cv_test_idx]=yhat_i
        cv_yhat_predict[r*n_splits+s,:]=cv_model.predict(X_predict)
        s+=1
   
    #print('cv_yhat_train', cv_yhat_train.shape, cv_yhat_train)

    
    CVPlusPI().run(y_train,cv_yhat_train,cv_yhat_predict,true_y_predict=true_y_predict)
    