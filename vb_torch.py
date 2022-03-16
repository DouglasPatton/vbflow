import torch.nn as nn
from torch.optim import SGD
from torch import Tensor,load,save,tanh

from io import BytesIO
from joblib import hash as jhash #because hash is reserved by python
from sklearn.datasets import make_regression
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import make_pipeline,Pipeline

from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures#,StandardScaler, FunctionTransformer, , OneHotEncoder, PowerTransformer,
from sklearn.model_selection import cross_validate, train_test_split, RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import TransformedTargetRegressor
#import matplotlib.pyplot as plt
from vb_transformers import logminplus1_T,shrinkBigKTransformer,dropConst#,logminus_T,exp_T,,none_T,log_T, logp1_T,dropConst,columnBestTransformer

from missing_val_transformer import missingValHandler

from vb_estimators import myLogger

class TorchRegressor(myLogger):
    # pytorch implementation inspired by: 
    ## https://docs.microsoft.com/en-us/archive/msdn-magazine/2019/march/test-run-neural-regression-using-pytorch
    
    def __init__(self,epochs=100,splitter=None,checkpoint=None,hidden_layers=1,nodes_per_layer=100,learning_rate=0.01,dropout_share=0.5,shuffle_by_epoch=True):
        myLogger.__init__(self,name='torchregressor.log')
        self.logger.info('starting TorchRegressor logger)')
        self.splitter=splitter
        self.checkpoint=checkpoint
        self.epochs=epochs
        self.learning_rate=learning_rate
        self.nodes_per_layer=nodes_per_layer
        self.hidden_layers=hidden_layers
        self.dropout_share=dropout_share
        self.shuffle_by_epoch=shuffle_by_epoch #for deterministic splitters repeated across epochs
        
        
    def get_params(self,deep=False):
        return dict(splitter=self.splitter,checkpoint=self.checkpoint,epochs=self.epochs,learning_rate=self.learning_rate,
                    nodes_per_layer=self.nodes_per_layer,hidden_layers=self.hidden_layers,dropout_share=self.dropout_share)
                    
    def set_params(self,p_dict,deep=False):
        self.splitter=p_dict['splitter']
        self.checkpoint=p_dict['checkpoint']
        self.epochs=p_dict['epochs']
        self.learning_rate=p_dict['learning_rate']
        self.nodes_per_layer=p_dict['nodes_per_layer']
        self.hidden_layers=p_dict['hidden_layers']
        self.dropout_share=p_dict['dropout_share']
        
    def fit(self,X,y=None,w=None):
        assert not y is None
        try:
            self.net_
        except:
        
            activation=lambda: nn.ReLU()#nn.Sigmoid()#nn.ReLU()#nn.LeakyReLU()#nn.ReLU()# 
            lin_depth=self.hidden_layers
            width=self.nodes_per_layer
            layers=[nn.Linear(X.shape[1],width),
                    nn.BatchNorm1d(width),
                    activation(),
                    nn.Dropout(self.dropout_share)]
            for _ in range(lin_depth):
                layers.extend(
                    [nn.Linear(width,width),nn.BatchNorm1d(width),activation(),nn.Dropout(self.dropout_share)])
            self.net_=nn.Sequential(*layers,nn.Linear(width,1))
            self.optimizer=SGD(self.net_.parameters(),lr=self.learning_rate,momentum=0.1)
            self.loss_func=nn.MSELoss()

            if not self.checkpoint is None:
                self.set_net(train=True)
        self.net_.train()
        i=0
        shuf_idx=np.arange(y.size)
        if self.shuffle_by_epoch: np.random.default_rng(seed=self.epochs).shuffle(shuf_idx)
        for split_idx in self.splitter.split(X,y):
            if type(split) is tuple:split_idx=split_idx[-1] #just using test_idx, so one round of cv is all the data
            self.optimizer.zero_grad()
            X_i=Tensor(X[shuf_idx][split_idx])
            y_i=Tensor(y[shuf_idx][split_idx]).view(-1,1)#makes into row matrix with 1 column and automatic/appropriate (-1) number of rows.
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
            self.epochs+=checkpoint_dict['epoch']
        
    def save_net(self,):
        checkpoint_dict={
            'model_state_dict':self.net_.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict(),
            'epoch':self.epochs,
            #'loss':loss
            }
        byte_stream=BytesIO()
        save(checkpoint_dict,byte_stream)
        byte_stream.seek(0)
        self.checkpoint=byte_stream.read()
            #https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html#save-the-general-checkpoint


class TorchNet(BaseEstimator,RegressorMixin,myLogger): 
    #
    def __init__(self,do_prep=True,prep_dict={'impute_strategy':'impute_knn5'},
                 epochs=1000,batch_size=128,polynomial_degree=3,cat_idx=None,float_idx=None,inner_cv=None,checkpoint=None):
        myLogger.__init__(self,name='torchnet.log')
        self.logger.info('starting torchnet logger')
        self.do_prep=do_prep
        self.inner_cv=inner_cv
        self.cat_idx=cat_idx
        self.float_idx=float_idx
        self.prep_dict=prep_dict
        self.epochs=epochs
        self.batch_size=batch_size
        self.checkpoint_start=checkpoint #used for creating models that have been fit already, 
        ##unlike self.checkpoint_, which stores the latest fit model (could be the start if 
        ##one is provided and self.checkpoint_ is still None
        self.fit_counter=0 # to allow refitting/partial_fit with more data...eventually
    
    def fit(self,X,y=None,w=None):
        self.n,self.k=X.shape
        assert not y is None
            
        try:
            self.checkpoint_
            self.fit_counter=self.checkpoint_['epoch'] 
            print(f'checkpoint found and fit_counter:{self.fit_counter}')
        except:
            if not self.checkpoint is None:
                self.checkpoint_=self.checkpoint_start
                self.checkpoint=None
            else:
                self.checkpoint_=None
                
        
        n_splits=self.n//self.batch_size
        if self.inner_cv is None:
            self.logger.info(f'setting inner_cv with n_splits:{n_splits}, n_repeats:{self.epochs}')
            self.inner_cv=RepeatedKFold(
                n_splits=n_splits,n_repeats=self.epochs,random_state=self.fit_counter)
        try: self.pipe_
        except:
            if self.checkpoint_:
                self.pipe_=self.get_pipe()
            else:
                self.pipe_=TorchTuner(self.get_pipe,self.tune_dict)#self.get_pipe() #uses self.checkpoint_ if it exists
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
        
    
    def get_pipe(self,options_dict={}):
        if not 'checkpoint' in options_dict:
            options_dict['checkpoint']=self.checkpoint_
        if not 'splitter' in options_dict:
            options_dict['splitter']=self.inner_cv
        try:
            degree=options_dict.pop('polynomial_degree')
        except:
            degree=3
        try:
            k_share=options_dict.pop('k_share')
        except:
            k_share=0.25
            
        
        #tlist=[logminplus1_T],#not implemented
        steps=[
            ('scaler',MinMaxScaler()),
            ('select',shrinkBigKTransformer(max_k=50)),
            ('polyfeat',PolynomialFeatures(interaction_only=0,degree=degree)),
            ('drop_constant',dropConst()),
            ('select2',shrinkBigKTransformer(k_share=k_share)),
            ('reg',TransformedTargetRegressor(TorchRegressor(**options_dict),transformer=MinMaxScaler()))]
        outerpipe=Pipeline(steps=steps)
        
        if self.do_prep:
            steps=[('prep',missingValHandler(prep_dict=self.prep_dict)),
                   ('post',outerpipe)]
            outerpipe=Pipeline(steps=steps)
        
        return outerpipe
    
class TorchTuner(BaseEstimator,RegressorMixin,myLogger):
    def __init__(self,pipe_caller,tune_dict=None,memory_option='model'):#other memory_option values: 'checkpoint','disk'
        myLogger.__init__(self,name='torchtuner')
        if tune_dict is None:
            tune_dict={'hidden_layers':[1,2,3],'nodes_per_layer'=100,'epochs'=256,'tune_rounds'=5,'polynomial_degree'=[2,3],'learning_rate'=[0.1,0.01]}
        self.tune_dict=tune_dict
        self.tune_rounds=tune_dict.pop('tune_rounds')
    def fit(self,X,y,w=None):
        self.results_={}
        self.setup_build_dicts(tune_dict)
        for r in range(tune_rounds):
            for build_dict in self.get_build_dict(r):
                self.my_cross_val_score(pipe_caller,build_dict,X,y,w,r)
            self.setup_next_round(r)
        return self.get_best_from results()
    
    def my_cross_val_score(pipe_caller,build_dict,X,y,w,r):
        if 'pipe_list' in build_dict:
            pipe_list=build_dict['pipe_list']
        else:
            cv_loss
            pipe_list=None#pipe_caller(**kwargs) for _ in range(self.inner_cv.get_n_splits())]
        for i,(train_idx,test_idx) in enumerate(self.inner_cv.split(X,y,w)):
            if not w is None:
                w_train=w[train_idx]
                w_test=w[test_idx]
            else:
                w_train=None
                w_test=None
            if not pipe_list is None:
                pipe_i=pipe_list[i]
            else:
                pipe_i=pipe_caller(**build_dict['kwargs'])
            pipe_i.fit(X[train_idx],y[train_idx],w=w_train)
            
                #pipe_list.append(pipe_i)
            yhat=pipe_i.predict(X[test_idx],y[test_idx],w=w_test)
            loss=self.calculate_loss(y,yhat)
            build_dict['cv_loss'].append((r,i,loss))
            if self.memory_option=='model' and r==0:
                build_dict['pipe_list'].append(pipe_i)

            
class TorchTunerResultsTool(myLogger):
    def __init__(self,memory_option,tune_dict,kwargs):
        myLogger.__init__(self,name='torch_tuner_results_tool.log')
        self.memory_option=memory_option
        self.tune_dict=tune_dict
        self.results_by_round=[]
        self.tune_dict_list=None
        self.r=0
                 
    def start_round_0(self,tune_dict):
        self.tune_dict_list=self.get_tune_dicts(tune_dict)
        
    
    def new_round():
        self.r+=1
        self.kwargs_list=self.get_next_results()
         
                    
    def next_kwargs():
        return self.kwargs_list.pop()
                    
    def add_result(self,result):
        self.results_by_round[-1].append(result)
        
    def get_tune_dicts(self,tune_dict):
        tune_dict_list=[]
        kwargs=dict.fromkeys(tune_dict)
        while tune_dict:
            for i,(key,val) in enumerate(tune_dict.items()):
                if type(val) is list and 
                    if len(val)>0:
                        kwargs[key]=val.pop()
                        if i>0:
                            tune_dict_list.append(kwargs.copy())
                            break
                    else:results_tool.tune_dict.pop(key)
                else:
                    kwargs[key]=val
                    tune_dict.pop(key)
                    if i>0:
                        tune_dict_list.append(kwargs.copy())
                        break
                if i==0:
                    tune_dict_list.append(kwargs.copy())
        return tune_dict_list


        
        
                        
                    
            
        
    
            
                   
    
                   
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