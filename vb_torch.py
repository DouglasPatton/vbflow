import pandas as pd
import torch.nn as nn
from torch.optim import SGD
from torch import Tensor,load,save,tanh,set_num_threads,no_grad
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



class TorchRegressor(BaseEstimator,RegressorMixin,myLogger):
    # pytorch implementation inspired by: 
    ## https://docs.microsoft.com/en-us/archive/msdn-magazine/2019/march/test-run-neural-regression-using-pytorch
    
    def __init__(self,epochs=500,splitter=None,checkpoint=None,hidden_layers=1,nodes_per_layer=100,momentum=0.1,learning_rate=0.01,dropout_share=0.1,shuffle_by_epoch=True,train_noise=False,batch_size=64,set_net_fit=False):
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
        self.set_net_fit=set_net_fit
        
        
    def get_params(self,deep=False):
        return dict(
            splitter=self.splitter,checkpoint=self.checkpoint,epochs=self.epochs,
            learning_rate=self.learning_rate,nodes_per_layer=self.nodes_per_layer,
            hidden_layers=self.hidden_layers,dropout_share=self.dropout_share,batch_size=self.batch_size,
            train_noise=self.train_noise,momentum=self.momentum,set_net_fit=self.set_net_fit)
                    
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
        self.set_net_fit=p_dict['set_net_fit']
        
    def fit(self,X,y=None,w=None):
        set_num_threads(4)
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
            if self.set_net_fit:
                assert not self.checkpoint is None
                self.set_net(train=False)
                self.checkpoint_=self.checkpoint
                return self
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
        self.checkpoint_=self.checkpoint
        #print(self.checkpoint_)
        return self

    def predict(self,X):
        try:
            self.net_
        except:
            assert not self.checkpoint is None,f'TorchRegressor model has not been trained yet!!!'
            self.set_net(train=False)
        self.net_.eval()
        with no_grad():
            return self.net_(Tensor(X)).detach().numpy()
            
            
    def set_net(self,train=False):
        try:
            #check_stream=BytesIO().write(self.checkpoint).seek(0)
            #checkpoint_dict=torch.load(check_stream)
            checkpoint_dict=self.checkpoint
            self.checkpoint_=self.checkpoint
            print('!!!!!!!!!!loading torch from state_dict!!!!!!!!!!!!!')
            self.net_.load_state_dict(checkpoint_dict['model_state_dict'])
            if train:
                assert self.set_net_fit is False
                self.optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
                self.epochs+=checkpoint_dict['epoch']
        except: 
            print(format_exc())
            print(checkpoint_dict)
            assert False
        
    def save_net(self,):
        self.net_.eval()
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
        #print(f'checkpoint saved. checkpoint_dict: {checkpoint_dict}')
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
                    'hidden_layers':[0,1],'nodes_per_layer':[80],
                    'epochs':[50],'tune_rounds':10,'cut_round':1,
                    'max_k':50,'polynomial_degree':[2,3],
                    'nystroem':False,'learning_rate':[0.01],
                    'momentum':[0.8],'batch_size':64,'train_noise':[True],'steps_per_round':1}
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

            if self.do_tune:
                
                if self.do_prep:
                    self.checkpoint_=self.pipe_['post'].best_model_[-1] #checkpoint list
                else:
                    self.checkpoint_=self.pipe_.best_model_[-1]
            else:
                if self.do_prep:
                    self.checkpoint_=self.pipe_['post']['reg'].regressor_.checkpoint
                else:
                    self.checkpoint_=self.pipe_['reg'].regressor_.checkpoint
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
            k_share=0.8
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
        
        
class AvgPipe(BaseEstimator,RegressorMixin):
    def __init__(self,pipe_list):
        self.pipe_list=pipe_list
    def fit(self,X,y=None):
        return self
    def predict(self,X):
        return np.concatenate([pipe.predict(X)[:,None] for pipe in self.pipe_list],axis=1).mean(axis=1)
        
    
class TorchTuner(BaseEstimator,RegressorMixin,myLogger):
    def __init__(self,pipe_caller,tune_dict,memory_option='model'):#other memory_option values: 'checkpoint','disk'
        myLogger.__init__(self,name='torchtuner')
        self.tune_dict=tune_dict
        self.pipe_caller=pipe_caller
        self.memory_option=memory_option
        self.keep_going=True
        
    def fit(self,X,y,w=None):
        
        self.best_model_=None
        self.tune_rounds_=self.tune_dict.pop('tune_rounds')
        self.cut_round_=self.tune_dict.pop('cut_round')
        self.steps_per_round=self.tune_dict.pop('steps_per_round')
        
        for r in range(self.tune_rounds_):
            if not self.keep_going:break
            self.setup_next_round(r)
            for build_dict in self.iter_build_dicts(r):
                self.my_cross_val_score(self.pipe_caller,build_dict,X,y,w,r)
        l,r,s,build_dict,checkpoint_list=self.best_model_
        #build_dict['kwargs']['epochs']=build_dict['kwargs']['epochs']*(r*self.steps_per_round+s)
        self.tuned_pipe_=AvgPipe(build_dict['pipe_list']).fit(X,y)
        #self.tuned_pipe_=self.pipe_caller(options_dict=build_dict['kwargs']).fit(X,y)
        #self.tuned_pipe_=self.merge_pipes(build_dict).fit(X,y)
        return self
    
    
    def merge_pipes(self,build_dict):
        assert 'pipe_list' in build_dict
        checkpoints=[pipe['reg'].regressor_.checkpoint_['model_state_dict'] for pipe in build_dict['pipe_list']]
        m=len(checkpoints)
        new_checkpoint=dict.fromkeys(checkpoints[0])
        
        for key in list(new_checkpoint.keys()):
            try:
                new_checkpoint[key]=1/m*sum([ch[key] for ch in checkpoints])
            except:
                print(key,[ch[key] for ch in checkpoints])
                assert False,format_exc()
        
        
        merged_pipe=self.pipe_caller(options_dict={**build_dict['kwargs'],'checkpoint':{'model_state_dict':new_checkpoint},'set_net_fit':True})
        
        return merged_pipe             
            
                     
                     
                         
                             
        
    def predict(self,X):
        try:
            yhat=self.tuned_pipe_.predict(X)
        except:
            self.logger.exception(f'error predicting. nan-count: {np.isnan(X).sum()} X:{X}')
          
            assert False
        return yhat
    
    def get_params(self,deep=False):
        p_dict={}
        p_dict['tune_dict']=self.tune_dict
        p_dict['pipe_caller']=self.pipe_caller
        p_dict['memory_option']=self.memory_option
        
        
    def set_params(self,p):
        self.tune_dict=p_dict['tune_dict']
        self.pipe_caller=p_dict['pipe_caller']
        self.memory_option=p_dict['memory_option']
        
        
    
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
            cols=['lss','rnd','step','cv_i']
            active_list=[]
            not_stopped_count=0
            imax=self.steps_per_round*r-1
            for b_idx,build_dict in enumerate(self.build_dicts_):
                if not build_dict['keep_training']:continue
                not_stopped_count+=1
                '''
                loss_df=pd.DataFrame(build_dict['cv_loss'],columns=cols)
                loss_df=loss_df[loss_df['rnd']>=r-1] #drop all but last 2 rounds
                avg_loss_df=loss_df.loc[:,cols[:-1]].groupby(['rnd','step']).mean()
                avg_loss_drop=avg_loss_df.sort_values(['rnd','step']).groupby(['rnd'])['lss'].diff()
                if avg_loss_drop'''
                
                
                cv_loss=[];cv_loss_lag=[];cv_loss_lag2=[]
                cv_loss_lag_dict={}
                for lss,rnd,step,cv_i,_ in build_dict['cv_loss']: #(loss,r,s,i,checkpoint)
                    #if rnd<r-3:continue
                    i=rnd*self.steps_per_round+step
                    if i<imax-self.steps_per_round*2:continue
                    if not (rnd,step) in cv_loss_lag_dict:
                        cv_loss_lag_dict[(rnd,step)]=[lss]
                    else:
                        cv_loss_lag_dict[(rnd,step)].append(lss)
                for key,val in cv_loss_lag_dict.items():
                    cv_loss_lag_dict[key]=sum(val)/len(val)
                iter_sorted=sorted([(key[0]*self.steps_per_round+key[1],key,val) for key,val in cv_loss_lag_dict.items()])
                if len(iter_sorted)>2 and iter_sorted[-1][-1]>iter_sorted[-2][-1] and iter_sorted[-1][-1]>iter_sorted[-3][-1]:
                    self.build_dicts_[b_idx]['keep_training']=False
                    #loss_dict_r.pop(b_idx)
                        #stopped_this_round.append(b_idx)
                else:
                    loss_sorted=sorted([(val,key) for key,val in cv_loss_lag_dict.items()])
                    active_list.append((*loss_sorted[0],b_idx))
                    
            losses=sorted(active_list)
            print(f'round: {r} losses: {losses}')
            if len(losses)==0:
                print(f'all models cut')
                self.keep_going=False
                return
            #l_,key_idx_list,b_idx_list=zip(*sorted(losses))
            if r>=self.cut_round_:
                if len(losses)>1: 
                    cut=-(-len(losses)//2)#ceiling divide
                    if len(losses)>cut:
                        for tup in losses[cut:]:
                            b_idx=tup[-1]
                            self.build_dicts_[b_idx]['keep_training']=False
                        #b_idx_list=b_idx_list[:int(len(b_idx_list)*0.5)]
                    #for b_idx in loss_dict_r.keys():
                    #    if not b_idx in b_idx_list:
                    #        print(f'dropping model {b_idx} with loss: {loss_dict_r[b_idx]}')
                    #        self.build_dicts_[b_idx]['keep_training']=False
            best_model_r_tup=losses[0]
            best_loss=best_model_r_tup[0]
            best_r,best_s=best_model_r_tup[1]
            best_idx=best_model_r_tup[2]
            _,best_checkpoints=zip(*sorted([(cv_i,tup[3]) for tup in self.build_dicts_[best_idx]['cv_loss'] if tup[1]==best_r and tup[2]==best_s]))
            if self.best_model_ is None:
                self.best_model_=(best_loss,best_r,best_s,deepcopy(self.build_dicts_[best_idx]),deepcopy(best_checkpoints))
            if best_loss<self.best_model_[0]:
                    self.best_model_=(best_loss,best_r,best_s,deepcopy(self.build_dicts_[best_idx]),deepcopy(best_checkpoints))
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
        print(f'r:{r}',end='. ')
        for i,(train_idx,test_idx) in enumerate(splitter.split(X,y,w)):
            #if i==3:break
            if not w is None:
                w_train=w[train_idx]
                w_test=w[test_idx]
            else:
                w_train=None
                w_test=None
            if not pipe_list is None:
                pipe_i=pipe_list[i]
            else:
                #print('creating new pipeline')
                pipe_i=pipe_caller(options_dict=build_dict['kwargs'])
            #print(f'starting fit cv i: {i}, in round: {r}.')
            print(f'{i}',end=',')
            xi=X[train_idx];yi=y[train_idx]
            xj=X[test_idx];yj=y[test_idx]
            for s in range(self.steps_per_round):     
                pipe_i.fit(xi,yi)
                #print(pipe_i['reg'].regressor.get_checkpoint())
                    #pipe_list.append(pipe_i)
                yhat=pipe_i.predict(xj)
                try:
                    loss=mean_squared_error(yj,yhat)
                except:
                    loss=99999
                build_dict['cv_loss'].append((loss,r,s,i,pipe_i['reg'].regressor_.checkpoint_))
            if self.memory_option=='model' and r==0:
                build_dict['pipe_list'].append(pipe_i)
        #print('bd_checkpoint',pipe_i['reg'].regressor_.checkpoint_)
                
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