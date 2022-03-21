import pandas as pd
import torch.nn as nn
from torch.optim import SGD
from torch import Tensor,load,save,tanh,set_num_threads
import numpy as np
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
from sklearn.kernel_approximation import Nystroem
from missing_val_transformer import missingValHandler
from traceback import format_exc
from copy import deepcopy
from vb_estimators import myLogger



class TorchRegressor(myLogger):
    # pytorch implementation inspired by: 
    ## https://docs.microsoft.com/en-us/archive/msdn-magazine/2019/march/test-run-neural-regression-using-pytorch
    
    def __init__(self,epochs=1500,splitter=None,checkpoint=None,hidden_layers=2,nodes_per_layer=100,momentum=0.1,learning_rate=0.01,dropout_share=0.1,shuffle_by_epoch=True,train_noise=False,batch_size=64,):
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
        self.train_noise=train_noise
        self.batch_size=batch_size
        self.momentum=momentum
        
        
    def get_params(self,deep=False):
        return dict(splitter=self.splitter,checkpoint=self.checkpoint,epochs=self.epochs,learning_rate=self.learning_rate,
                    nodes_per_layer=self.nodes_per_layer,hidden_layers=self.hidden_layers,dropout_share=self.dropout_share,batch_size=self.batch_size,train_noise=self.train_noise,momentum=self.momentum)
                    
    def set_params(self,p_dict,deep=False):
        self.splitter=p_dict['splitter']
        self.checkpoint=p_dict['checkpoint']
        self.epochs=p_dict['epochs']
        self.learning_rate=p_dict['learning_rate']
        self.nodes_per_layer=p_dict['nodes_per_layer']
        self.hidden_layers=p_dict['hidden_layers']
        self.dropout_share=p_dict['dropout_share']
        self.batch_size=p_dict['batch_size']
        self.train_noise=p_dict['train_noise']
        self.momentum=p_dict['momentum']
        
    def fit(self,X,y=None,w=None):
        set_num_threads(15)
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
            self.optimizer=SGD(self.net_.parameters(),lr=self.learning_rate,momentum=self.momentum)
            self.loss_func=nn.MSELoss()

            if not self.checkpoint is None:
                self.set_net(train=True)
        if self.train_noise:
            rng=np.random.default_rng(seed=self.epochs)
            X+=rng.standard_normal(size=X.shape)/10/self.epochs**0.5
            y+=rng.standard_normal(size=y.shape)/10/self.epochs**0.5
        self.net_.train()
        shuf_idx=np.arange(y.size)
        epochs_left=self.epochs
        if self.splitter is None:
            splitter=RepeatedKFold(
            n_splits=y.size//self.batch_size,n_repeats=1,random_state=self.epochs)
        else:splitter=self.splitter
        if self.shuffle_by_epoch:
            rng=np.random.default_rng(seed=self.epochs)
        while epochs_left:
            if self.shuffle_by_epoch: rng.shuffle(shuf_idx)
            X=X[shuf_idx,:]
            y=y[shuf_idx]
            epochs_left-=1
            for split_idx in splitter.split(X,y):
                
                if type(split_idx) is tuple:split_idx=split_idx[-1] #just using test_idx, so one round of cv is all the data
                self.optimizer.zero_grad()
                X_i=Tensor(X[split_idx,:])
                y_i=Tensor(y[split_idx]).view(-1,1)#makes into row matrix with 1 column and automatic/appropriate (-1) number of rows.
                loss_obj=self.loss_func(self.net_(X_i),y_i)
                loss_obj.backward()
                self.optimizer.step()
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
        #check_stream=BytesIO().write(self.checkpoint).seek(0)
        #checkpoint_dict=torch.load(check_stream)
        checkpoint_dict=self.checkpoint
        print('!!!!!!!!!!loading torch from state_dict!!!!!!!!!!!!!')
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
        #byte_stream=BytesIO()
        #save(checkpoint_dict,byte_stream)
        #byte_stream.seek(0)
        #self.checkpoint=byte_stream.read()
            #https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html#save-the-general-checkpoint
        self.checkpoint=checkpoint_dict


class TorchNet(BaseEstimator,RegressorMixin,myLogger): 
    #
    def __init__(self,do_prep=True,prep_dict={'impute_strategy':'impute_knn5'},
                 epochs=1000,batch_size=32,cat_idx=None,float_idx=None,inner_cv=None,checkpoint_start=None,tune_dict=None,do_tune=True):
        myLogger.__init__(self,name='torchnet.log')
        
        self.tune_dict=tune_dict
        self.do_tune=do_tune
        self.logger.info('starting torchnet logger')
        self.do_prep=do_prep
        self.inner_cv=inner_cv
        self.cat_idx=cat_idx
        self.float_idx=float_idx
        self.prep_dict=prep_dict
        self.epochs=epochs
        self.batch_size=batch_size
        #self.checkpoint=None
        self.checkpoint_start=checkpoint_start #used for creating models that have been fit already, 
        ##unlike self.checkpoint_, which stores the latest fit model (could be the start if 
        ##one is provided and self.checkpoint_ is still None
        self.fit_counter=0 # to allow refitting/partial_fit with more data...eventually
    
    def fit(self,X,y=None,w=None):
        try:
            if self.tune_dict is None:
                self.tune_dict={
                    'hidden_layers':[1,2],'nodes_per_layer':[100],
                    'epochs':50,'tune_rounds':15,'cut_round':2,
                    'max_k':[8,64],'polynomial_degree':[1,2],
                    'nystroem':False,'learning_rate':[0.01],
                    'momentum':[0.2,],'batch_size':64,'train_noise':[True,False]}
            self.n,self.k=X.shape
            assert not y is None

            try:
                self.checkpoint_
                self.fit_counter=self.checkpoint_['epoch'] 
                print(f'checkpoint found and fit_counter:{self.fit_counter}')
            except AttributeError:
                if not self.checkpoint_start is None:
                    self.checkpoint_=self.checkpoint_start
                    self.checkpoint_start=None
                else:
                    self.checkpoint_=None
            except: assert False, format_exc()


            try: self.pipe_.fit(X,y)
            except AttributeError:
                self.pipe_=self.get_pipe()
                self.pipe_.fit(X,y)
            except: assert False,format_exc()


            if self.do_prep:
                self.checkpoint_=self.pipe_['post'].tuned_pipe_['reg'].regressor_.checkpoint
            else:
                self.checkpoint_=self.pipe_['reg'].regressor_.tuned_pipe_.checkpoint
            return self
        except:
            self.logger.exception(f'outer catch')
            
    def predict(self,X,y=None,w=None):
        try:
            self.pipe_
        except:
            print('pipe not found')
            assert not self.checkpoint_ is None
            self.pipe_=self.get_pipe()
        return self.pipe_.predict(X)
        
    
    def get_pipe(self,options_dict={}):
        if self.do_tune:
            outerpipe=TorchTuner(self.get_post_pipe,self.tune_dict)
        else:
            outerpipe=self.get_post_pipe(options_dict=options_dict)       
        
        if self.do_prep:
            steps=[('prep',missingValHandler(prep_dict=self.prep_dict)),
                   ('post',outerpipe)]
            outerpipe=Pipeline(steps=steps)
        
        return outerpipe
    
    
    def get_post_pipe(self,options_dict={}):
        
        if not 'checkpoint' in options_dict:
            options_dict['checkpoint']=self.checkpoint_
        if not 'splitter' in options_dict:
            options_dict['splitter']=self.inner_cv
        try:
            degree=options_dict.pop('polynomial_degree')
        except:
            degree=2
        try:
            k_share=options_dict.pop('k_share')
        except:
            k_share=0.5
        try:
            max_k=options_dict.pop('max_k')
        except:
            max_k=50
        try:
            if options_dict['nystroem']:
                f_transform=Nystroem(random_state=0)
            else: assert False,'triggering polyfeatures bc nystroem is False'
        except:
            f_transform=PolynomialFeatures(interaction_only=0,degree=degree)
        if 'nystroem' in options_dict:options_dict.pop('nystroem')
                
            
        
        #tlist=[logminplus1_T],#not implemented
        steps=[
            ('scaler',MinMaxScaler()),
            ('select',shrinkBigKTransformer(max_k=max_k)),
            ('polyfeat',f_transform),
            ('drop_constant',dropConst()),
            ('select2',shrinkBigKTransformer(k_share=k_share)),
            ('reg',TransformedTargetRegressor(TorchRegressor(**options_dict),transformer=MinMaxScaler()))]
        outerpipe=Pipeline(steps=steps)
        
        return outerpipe
        
        
        
        
    
class TorchTuner(BaseEstimator,RegressorMixin,myLogger):
    def __init__(self,pipe_caller,tune_dict,memory_option='model'):#other memory_option values: 'checkpoint','disk'
        myLogger.__init__(self,name='torchtuner')
        self.tune_dict=tune_dict
        self.pipe_caller=pipe_caller
        self.memory_option=memory_option
    def fit(self,X,y,w=None):
        self.tune_rounds_=self.tune_dict.pop('tune_rounds')
        self.cut_round_=self.tune_dict.pop('cut_round')
        for r in range(self.tune_rounds_):
            self.setup_next_round(r)
            for build_dict in self.iter_build_dicts(r):
                self.my_cross_val_score(self.pipe_caller,build_dict,X,y,w,r)
        l,r,build_dict=self.best_model_
        build_dict['kwargs']['epochs']=build_dict['kwargs']['epochs']*r
        self.tuned_pipe_=self.pipe_caller(options_dict=build_dict['kwargs']).fit(X,y)
        return self.tuned_pipe_
        
    def predict(self,X):
        return self.tuned_pipe_.predict(X)
    
    def get_params(self):
        p_dict={}
        p_dict['tune_dict']=self.tune_dict
        p_dict['pipe_caller']=self.pipe_caller
        p_dict['memory_option']=self.memory_option
        
        
    def set_params(self,p):
        self.tune_dict=p_dict['tune_dict']
        self.pipe_caller=p_dict['pipe_caller']
        self.memory_option=p_dict['memory_option']
        
    def get_best_pipe():
        loss,r,build_dict=self.best_model_
        
    
    def iter_build_dicts(self,r):
        for build_dict in self.build_dicts_:
            if build_dict['keep_training']:
                yield build_dict
    
    def setup_next_round(self,r):
        print(f'TorchTuner setting up round: {r}.')
        if r==0:
            self.build_dicts_=self.setup_build_dicts()
        else:
            #stopped_this_round=[]
            loss_dict_r={}
            loss_dict_r_lag={}
            loss_dict_r_lag2={}
            for b_idx,build_dict in enumerate(self.build_dicts_):
                if not build_dict['keep_training']:continue
                cv_loss=[];cv_loss_lag=[];cv_loss_lag2=[]
                for rnd,cv_i,lss in build_dict['cv_loss']:
                    if rnd==r-1:
                        cv_loss.append(lss)
                    if rnd==r-2:
                        cv_loss_lag.append(lss)
                    if rnd==r-3:
                        cv_loss_lag2.append(lss)
                loss_dict_r[b_idx]=sum(cv_loss)/len(cv_loss)
                if r>2:
                    loss_dict_r_lag[b_idx]=sum(cv_loss_lag)/len(cv_loss_lag)
                    loss_dict_r_lag2[b_idx]=sum(cv_loss_lag2)/len(cv_loss_lag2)
            if r>2:#stop training models that are getting worse
                for b_idx in list(loss_dict_r.keys()):#list b/c modifying the dict
                    if loss_dict_r[b_idx]>loss_dict_r_lag[b_idx] and loss_dict_r_lag[b_idx]>loss_dict_r_lag2[b_idx]:
                        #print(f'dropping b_idx:{b_idx} with rising loss: {loss_dict_r[b_idx]}>{loss_dict_r_lag[b_idx]}')
                        self.build_dicts_[b_idx]['keep_training']=False
                        loss_dict_r.pop(b_idx)
                        #stopped_this_round.append(b_idx)
                    
            
            losses=[(loss,b_idx) for b_idx,loss in loss_dict_r.items()]
            print(f'round: {r} losses: {losses}')
            if len(losses)==0:
                print(f'all models cut')
                return
            l_,b_idx_list=zip(*sorted(losses))
            if r>=self.cut_round_:
                if len(b_idx_list)>1: 
                    b_idx_list=b_idx_list[:int(len(b_idx_list)*0.9)]
                for b_idx in range(len(self.build_dicts_)):
                    if not b_idx in b_idx_list:
                        self.build_dicts_[b_idx]['keep_training']=False
            if r==1:
                self.best_model_=(l_[0],r,self.build_dicts_[b_idx_list[0]].copy())
            if l_[0]<self.best_model_[0]:
                    self.best_model_=(l_[0],r,self.build_dicts_[b_idx_list[0]].copy())
        if r>0:print(f'tt has setup round : {r} and the best model loss so far is: {self.best_model_[0]}')
                    
            
                        
                    
    
    def my_cross_val_score(self,pipe_caller,build_dict,X,y,w,r):
        if type(X) is pd.DataFrame:X=X.to_numpy()
        if type(y) in (pd.DataFrame,pd.Series): y=y.to_numpy()
        if r==0 and self.memory_option=='model':
            build_dict['pipe_list']=[]
        if r>0 and 'pipe_list' in build_dict:
            pipe_list=build_dict['pipe_list']
        else:
            pipe_list=None#pipe_caller(**kwargs) for _ in range(self.inner_cv.get_n_splits())]
        splitter=RepeatedKFold(
            n_splits=5,n_repeats=1,random_state=r)
        for i,(train_idx,test_idx) in enumerate(splitter.split(X,y,w)):
            if i==2:break
            if not w is None:
                w_train=w[train_idx]
                w_test=w[test_idx]
            else:
                w_train=None
                w_test=None
            if not pipe_list is None:
                pipe_i=pipe_list[i]
            else:
                print('creating new pipeline')
                pipe_i=pipe_caller(options_dict=build_dict['kwargs'])
            print(f'starting fit cv i: {i}, in round: {r}.')
            pipe_i.fit(X[train_idx],y[train_idx])
            
                #pipe_list.append(pipe_i)
            yhat=pipe_i.predict(X[test_idx])
            try:
                loss=mean_squared_error(y[test_idx],yhat)
            except:
                loss=99999
            build_dict['cv_loss'].append((r,i,loss))
            if self.memory_option=='model' and r==0:
                build_dict['pipe_list'].append(pipe_i)
                
    def setup_build_dicts(self):
        tune_dict=deepcopy(self.tune_dict)
        build_dict_list=[]
        kwargs=dict.fromkeys(tune_dict) #kwargs takes on values in tune_dict, changing 
        ##one at a time (after the first round sets a val for each key).
        tracking_kwargs={'cv_loss':[],'keep_training':True}
        drop_keys=[]
        for key,val in tune_dict.items():
            if type(val) is list:
                if len(val)==0:
                    drop_keys.append(key)
                elif len(val)==1:
                    kwargs[key]=val[0]
                    drop_keys.append(key)
                else:
                    kwargs[key]=val.pop()
            else:
                kwargs[key]=val
                drop_keys.append(key)
        for key in drop_keys:tune_dict.pop(key)
        build_dicts_=[{'kwargs':deepcopy(kwargs),**deepcopy(tracking_kwargs)}]
        for key,vals in tune_dict.items():
            new_builds=[]
            for val in vals:
                for b_idx in range(len(build_dicts_)):
                    new_dict=deepcopy(build_dicts_[b_idx])
                    new_dict['kwargs'][key]=val
                    new_builds.append(new_dict)
            build_dicts_.extend(new_builds)
        return build_dicts_
                
                
            
            
        
   


        
        
                        
                    
            
        
    
            
                   
    
                   
if __name__=="__main__":
    import numpy as np
    X, y= make_regression(n_samples=500,n_features=6,noise=5)
    y=np.exp(y/(y.max()))*100
    #y=y**2
    net_pipeline=TorchNet(do_prep=True)
    net_pipeline.fit(X[:300],y[:300])
    #print(net_pipeline.predict(X),y)
    s=net_pipeline.score(X[-200:],y[-200:])
    print(f'r2 score: {s}')
    #cv=cross_validate(net_pipeline,X,y,scoring='r2',cv=2)
    #print(cv)
    #print(cv['test_score'])