import torch.nn as nn
from torch.optim import SGD
from torch import Tensor,load,save,tanh

from io import BytesIO

from sklearn.datasets import make_regression
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import make_pipeline,Pipeline

from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures#,StandardScaler, FunctionTransformer, , OneHotEncoder, PowerTransformer,
from sklearn.model_selection import cross_validate, train_test_split, RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import TransformedTargetRegressor
#import matplotlib.pyplot as plt
from vb_transformers import logminplus1_T,shrinkBigKTransformer#,logminus_T,exp_T,,none_T,log_T, logp1_T,dropConst,columnBestTransformer

from missing_val_transformer import missingValHandler

from vb_estimators import myLogger

class TorchRegressor(myLogger):
    
    def __init__(self,n_epochs,splitter,checkpoint=None):
        myLogger.__init__(self,name='torchregressor.log')
        self.logger.info('starting TorchRegressor logger)')
        self.splitter=splitter
        self.checkpoint=checkpoint
        self.n_epochs=n_epochs
        
    def get_params(self,deep=False):
        return dict(splitter=self.splitter,checkpoint=self.checkpoint,n_epochs=self.n_epochs)
                    
    def set_params(self,p_dict,deep=False):
        self.splitter=p_dict['splitter']
        self.checkpoint=p_dict['checkpoint']
        self.n_epochs=p_dict['n_epochs']
        
    def fit(self,X,y=None,w=None):
        assert not y is None
        activation=lambda: nn.ReLU()#nn.Sigmoid()#nn.ReLU()#nn.LeakyReLU()#nn.ReLU()# 
        lin_depth=1
        width=200
        layers=[nn.Linear(X.shape[1],width),
                nn.BatchNorm1d(width),
                activation(),
                nn.Dropout(0.7)]
        for _ in range(lin_depth):
            layers.extend(
                [nn.Linear(width,width),nn.BatchNorm1d(width),activation(),nn.Dropout(0.5)])
        self.net_=nn.Sequential(*layers,nn.Linear(width,1))
        self.optimizer=SGD(self.net_.parameters(),lr=0.001,momentum=0.9)
        self.loss_func=nn.MSELoss()
        
        if not self.checkpoint is None:
            self.set_net(train=True)
        self.net_.train()
        i=0
        for train_x,train_y in self.splitter.split(X,y):
            self.optimizer.zero_grad()
            X_i=Tensor(X[train_x])
            y_i=Tensor(y[train_x]).view(-1,1)
            loss_obj=self.loss_func(self.net_(X_i),y_i)
            loss_obj.backward()
            self.optimizer.step()
            i+=1
            #if True:#i%99==0:
            #    print(f'split {i}',end=',')
            #    self.logger.error(f'repetition {i}')
                
        self.save_net()
        
    def predict(self,X):
        try:
            self.net_
        except:
            assert not self.checkpoint is None,f'TorchRegressor model has not been trained yet!!!'
            self.set_net(train=False)
        self.net_.eval()
        
        return self.net_(Tensor(X)).detach().numpy()
            
            
    def set_net(self,train=False):
        check_stream=BytesIO().write(self.checkpoint).seek(0)
        
        checkpoint_dict=torch.load(check_stream)
        self.net_.load_state_dict(checkpoint_dict['model_state_dict'])
        if train:
            self.optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
            self.n_epochs+=checkpoint_dict['epoch']
        
    def save_net(self,):
        checkpoint_dict={
            'model_state_dict':self.net_.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict(),
            'epoch':self.n_epochs,
            #'loss':loss
            }
        byte_stream=BytesIO()
        save(checkpoint_dict,byte_stream)
        byte_stream.seek(0)
        self.checkpoint=byte_stream.read()
            #https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html#save-the-general-checkpoint


class TorchNet(BaseEstimator,RegressorMixin,myLogger): 
    # pytorch implementation inspired by: 
    # https://docs.microsoft.com/en-us/archive/msdn-magazine/2019/march/test-run-neural-regression-using-pytorch
    def __init__(self,do_prep=True,prep_dict={'impute_strategy':'impute_knn5'},
                 epochs=1000,batch_size=128,cat_idx=None,float_idx=None,inner_cv=None,checkpoint=None):
        myLogger.__init__(self,name='torchnet.log')
        self.logger.info('starting torchnet logger')
        self.do_prep=do_prep
        self.inner_cv=inner_cv
        self.cat_idx=cat_idx
        self.float_idx=float_idx
        self.prep_dict=prep_dict
        self.epochs=epochs
        self.batch_size=batch_size
        self.checkpoint=checkpoint
        self.fit_counter=0 # to allow refitting/partial_fit with more data...eventually
        #self.pipe_=self.get_pipe() #formerly inside basehelper         
        #BaseHelper.__init__(self) #TODO: create basehelper for pytorch neural nets
    
    def fit(self,X,y=None,w=None):
        self.n,self.k=X.shape
        assert not y is None
            
        try:
            self.checkpoint_
            self.fit_counter=self.checkpoint_['epoch'] 
            print(f'checkpoint found and fit_counter:{self.fit_counter}')
        except:
            if not self.checkpoint is None:
                self.checkpoint_=self.checkpoint
                self.checkpoint=None
            else:
                self.checkpoint_=None
                
        
        n_splits=self.n//self.batch_size
        if self.inner_cv is None:
            self.logger.info(f'setting inner_cv with n_splits:{n_splits}, n_repeats:{self.epochs}')
            self.inner_cv=RepeatedKFold(
                n_splits=n_splits,n_repeats=self.epochs,random_state=self.fit_counter)
            
        self.pipe_=self.get_pipe()
        self.pipe_.fit(X,y=y)

        if self.do_prep:
            self.checkpoint_=self.pipe_['post']['reg'].regressor_.checkpoint
        else:
            self.checkpoint_=self.pipe_['reg'].regressor_.checkpoint
        return self
            
    def predict(self,X,y=None,w=None):
        try:
            self.pipe_
        except:
            self.pipe_=self.get_pipe()
        return self.pipe_.predict(X)
        
    
    def get_pipe(self,):
        tlist=[logminplus1_T],#not implemented
        steps=[
            ('scaler',MinMaxScaler()),
            ('select',shrinkBigKTransformer(max_k=50)),
            ('polyfeat',PolynomialFeatures(interaction_only=0,degree=3)),
            ('select2',shrinkBigKTransformer(k_share=0.2)),
            ('reg',TransformedTargetRegressor(TorchRegressor(self.epochs,self.inner_cv,checkpoint=self.checkpoint_),transformer=MinMaxScaler()))]
        outerpipe=Pipeline(steps=steps)
        
        if self.do_prep:
            steps=[('prep',missingValHandler(prep_dict=self.prep_dict)),
                   ('post',outerpipe)]
            outerpipe=Pipeline(steps=steps)
        return outerpipe
                   
    
                   
if __name__=="__main__":
    import numpy as np
    X, y= make_regression(n_samples=300,n_features=12,noise=10)
    y=np.exp(y/(y.max()))*100
    net_pipeline=TorchNet(do_prep=True)
    net_pipeline.fit(X,y)
    print(net_pipeline.predict(X),y)
    s=net_pipeline.score(X,y)
    print(f'r2 score: {s}')
    #cv=cross_validate(net_pipeline,X,y,scoring='r2',cv=2)
    #print(cv)
    #print(cv['test_score'])