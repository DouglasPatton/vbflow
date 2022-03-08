import torch.nn as nn
from torch.optim import SGD
from torch import Tensor,load


from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import make_pipeline,Pipeline

from sklearn.preprocessing import MinMaxScaler#,StandardScaler, FunctionTransformer, PolynomialFeatures, OneHotEncoder, PowerTransformer,
from sklearn.model_selection import cross_validate, train_test_split, RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import TransformedTargetRegressor
import matplotlib.pyplot as plt
from vb_transformers import shrinkBigKTransformer,logminus_T,exp_T,logminplus1_T,none_T,log_T, logp1_T,dropConst,columnBestTransformer
from missing_val_transformer import missingValHandler

from vb_estimators import myLogger

class TorchRegressor(myLogger):
    
    def __init__(self,n_epochs,splitter,col_count,checkpoint=None):
        self.splitter=splitter
        self.col_count=col_count
        self.checkpoint=checkpoint
        self.n_epochs
        
    def fit(self,X,y=None,w=None):
        assert not y is None
        self.net=nn.Sequential(
                nn.Linear(self.k,10),
                nn.LeakyReLu(),
                nn.Linear(10,10),
                nn.LeakyReLu(),
                nn.Dropout()
                nn.Linear(10,1))
        self.optimizer=SGD(self.net.parameters(),lr=0.01)
        self.loss_func=nn.MSELoss()
        
        if not self.checkpoint is None:
            self.set_net(train=True)
        self.net.train()
        
        for train_x,train_y in self.splitter.split(X,y):
            self.optimizer.zero_grad()
            X_i=Tensor(X[train_x])
            y_i=Tensor(y[train_x]).view(-1,1)
            loss_obj=self.loss_func(net(X_i),y_i)
            loss_obj.backward()
            self.optimizer.step()
        self.save_net()
        
    def predict(self,X):
        try:
            self.net
        except:
            assert not self.checkpoint is None,f'TorchRegressor model has not been trained yet!!!'
            self.set_net(train=False)
        self.net.eval()
        
        return self.net(Tensor(X)).numpy()
            
            
    def set_net(self,train=False):
        checkpoint_dict=torch.load(self.checkpoint)
        self.net.load_state_dict(checkpoint_dict['model_state_dict'])
        if train:
            self.optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
            self.n_epochs+=checkpoint_dict['epoch']
        
    def save_net(self,):
        checkpoint_dict={
            'model_state_dict':self.net.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict()
            'epoch':self.n_epochs,
            #'loss':loss
            }
        self.checkpoint=torch.save(checkpoint_dict,'')
            #https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html#save-the-general-checkpoint


class TorchNet(BaseEstimator,RegressorMixin,myLogger): 
    # pytorch implementation inspired by: 
    # https://docs.microsoft.com/en-us/archive/msdn-magazine/2019/march/test-run-neural-regression-using-pytorch
    def __init__(self,do_prep=True,prep_dict={'impute_strategy':'impute_knn5'},
                 epochs=100,batch_size=32,cat_idx=None,float_idx=None,inner_cv=None,checkpoint=None):
        myLogger.__init__(self,name='torchnet.log')
        self.logger.info('starting torchnet logger')
        self.do_prep=do_prep
        self.inner_cv=inner_cv
        self.groupcount=groupcount #not used, can be deleted
        self.bestT=bestT
        self.cat_idx=cat_idx
        self.float_idx=float_idx
        self.prep_dict=prep_dict
        self.epochs=epochs
        self.checkpoint=None
        self.batch_size=batch_size
        self.fit_counter=0 # to allow refitting/partial_fit with more data...eventually
        #self.pipe_=self.get_pipe() #formerly inside basehelper         
        #BaseHelper.__init__(self) #TODO: create basehelper for pytorch neural nets
    
    def fit(self,X,y=None,w=None):
        self.n,self.k=X.shape
        assert not y is None
        
        if not self.checkpoint is None:
            self.fit_counter=self.checkpoint['epoch'] 
            
        n_splits=self.n//self.batch_size
        if self.inner_cv is None:
            self.inner_cv=RepeatedKFold(
                n_splits=n_splits,n_repeats=self.epochs,random_state=self.fit_counter)
            
        self.pipe_=self.get_pipe()
            
            
            
        self.pipe_.fit(X,y=y,w=w)
        if self.do_prep:
            self.checkpoint=self.pipe_['post']['reg'].checkpoint
        else:
            self.checkpoint=self.pipe_['reg'].checkpoint
            
    def predict(self,X,y=None,w=None):
        try:
            self.pipe_
        except:
            self.pipe_=self.get_pipe()
        self.pipe_.eval()
        
    
    def get_pipe(self,):
        
        steps=[
            ('scaler',MinMaxScaler()),
            ('reg',TorchRegressor(self.epochs,self.inner_cv,self.k,checkpoint=self.checkpoint))]
        outerpipe=Pipeline(steps=steps)
        
        if self.do_prep:
            steps=[('prep',missingValHandler(prep_dict=self.prep_dict)),
                   ('post',outerpipe)]
            outerpipe=Pipeline(steps=steps)
        return outerpipe
                   
    
                   
if __name__=="__main__":
    X, y= make_regression(n_samples=300,n_features=5,noise=1)
    net_pipeline=TorchNet(do_prep=True)
    net_pipeline.fit(X,y)
    s=net_pipeline.score(X,y)
    print(f'r2 score: {s}')
    cv=cross_validate(lrs,X,y,scoring='r2',cv=2)
    print(cv)
    print(cv['test_score'])