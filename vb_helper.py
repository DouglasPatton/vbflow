import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import ElasticNetCV, LinearRegression, Lars
from sklearn.base import BaseEstimator, TransformerMixin

class VBHelper:
    def __init__(self,test_share,cv_folds,cv_reps,cv_count,rs):
        self.test_share=test_share
        self.cv_folds=cv_folds
        self.cv_reps=cv_reps
        self.cv_count=cv_count
        self.rs=rs
        self.scorer_list=None
        self.max_k=None
        self.estimator_dict=None

        
    
    def plotCVScores(self,cv_score_dict,sort=1):
        colors = ['r', 'g', 'b', 'm', 'c', 'y', 'k']    
        fig=plt.figure(figsize=[12,15])
        plt.suptitle(f"Model Scores Across {self.cv_count} Cross Validation Runs. ")
        s_count=len(self.scorer_list)
        xidx=np.arange(self.cv_count) # place holder for scatterplot

        for s_idx, scorer in enumerate(self.scorer_list):
            ax=fig.add_subplot(f'{s_count}1{s_idx}')
            #ax.set_xlabel('estimator')
            #ax.set_ylabel(scorer)
            ax.set_title(scorer)
            for e_idx,estimator_name in enumerate(cv_score_dict.keys()):
                scores=cv_score_dict[estimator_name][scorer]
                if sort: scores.sort()
                ax.plot(xidx,scores,color=colors[e_idx],alpha=0.5,label=estimator_name+' cv score='+str(np.mean(cv_score_dict[estimator_name][scorer])))
                #ax.hist(scores,density=1,color=colors[e_idx],alpha=0.5,label=estimator_name+' cv score='+str(np.mean(cv_score_dict[estimator_name][scorer])))
            ax.grid(True)
            ax.xaxis.set_ticks([])
            ax.xaxis.set_visible(False)
            ax.legend(loc=4)
            fig.show()
            
            
    def printTestandCVScores(self, estimator_name,cv_score_dict_means):
        model=estimator_dict[estimator_name]()
        model.fit(X_train,y_train)
        if test_share:
            y_test_hat=model.predict(X_test)
            print(f'test set: negative-mse={-mean_squared_error(y_test,y_test_hat)}')
        for scorer in self.scorer_list:
            print(f'cv avg: {scorer}= {cv_score_dict_means[estimator_name][scorer]}')
            
  


class shrinkBigKTransformer(BaseEstimator,TransformerMixin):
    def __init__(self,max_k=500):
        self.max_k=max_k
        
        
    def fit(self,X,y):
        assert not y is None,f'y:{y}'
        self.n_samples_fit_ = X.shape[0]
        k=X.shape[1]
        if k>self.max_k:
            lars=Lars(fit_intercept=1,normalize=1,n_nonzero_coefs=self.max_k) #the magic
            lars.fit(X,y)
            self.col_select=np.arange(k)[lars.coef_>0]
        else:
            self.col_select=np.arange(k)
        return self
    
    def transform(self,X):
        n_samples_transform = X.shape[0]
        return X[:,self.col_select]
            

from sklearn.compose import TransformedTargetRegressor

class MultiCV(BaseEstimator,TransformerMixin):
    def __init__(self,max_k=500,estimator=None,transform_dict=None,scorer='neg_mean_squared_error'):
        self.max_k=max_k
        if estimator is None:
            assert False, 'not developed'
        self.estimator=estimator
        if transform_dict is None:
            transform_dict={'none':{'func':lambda x: x,
                                  'inverse_func': lambda x: x},
                            'log':{'func':lambda x: np.log(x0),
                                  'inverse_func': lambda x: np.exp(x)}}
        self.transform_dict=transform_dict
        
    def fit(self,X,y=None):
        for t_name,transform in transform_dict.items():
            TransformedTargetRegressor(regressor=estimator,**transform)
            TPipe=make_pipeline(transform,estimator)
        
        
    
        
    