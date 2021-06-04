from vb_helper import myLogger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import spearmanr,pearsonr
from scipy.cluster import hierarchy
import re

class VBSummary(myLogger):
    def __init__(self):
        myLogger.__init__(self)
        plt.rcParams['font.size'] = '8'
        
    def setData(self,df_dict):
        #data looks like: summary_data={'full_float_X':X_json_s,'full_y':y_json_s, 'X_nan_bool':X_nan_bool_s} 
        self.full_X_float_df=pd.read_json(df_dict['full_float_X'])
        self.full_y_df=pd.read_json(df_dict['full_y'])
        self.X_nan_bool_df=pd.read_json(df_dict['X_nan_bool'])
        
    def kernelDensity(self):
        all_vars=self.full_X_float_df.columns.to_list()
        float_vars=[name for name in all_vars if not re.search('__',name)]
        cat_vars=[name for name in all_vars if not name in float_vars]
        cat_var_dict=self.mergeCatVars(cat_vars)
        cat_group_names=list(cat_var_dict.keys())
        float_var_count=len(float_vars)
        total_var_count=float_var_count+len(cat_var_dict)+1 #dep var too
        
        plot_cols=int(total_var_count**0.5)
        plot_rows=-(-total_var_count//plot_cols) #ceiling divide
        fig,axes_tup=plt.subplots(nrows=plot_rows,ncols=plot_cols,figsize=(12,12),dpi=200)
        axes_list=[ax for axes in axes_tup for ax in axes]
        
        
        
        for ax_idx,ax in enumerate(axes_list):
            if ax_idx<float_var_count+1:
                if ax_idx==0:
                    self.full_y_df.plot.density(ax=ax,c='r')
                else:
                    name=float_vars[ax_idx-1]
                    self.full_X_float_df.loc[:,[name]].plot.density(ax=ax)
                    ax.legend(loc=9)
            elif ax_idx<total_var_count:
                cat_idx=ax_idx-float_var_count-1
                cat_name=cat_group_names[cat_idx]
                cat_flavors,var_names=zip(*cat_var_dict[cat_name])
                cat_df=self.full_X_float_df.loc[:,var_names]
                cat_df.columns=cat_flavors
                cat_shares=cat_df.sum()
                cat_shares.name=cat_name
                self.cat_shares=cat_shares
                cat_shares.plot(y=cat_name,ax=ax,kind='pie')
            else:
                ax.axis('off')
        fig.tight_layout()
        
        
    def mergeCatVars(self,var_names):
        var_dict={}
        for var in var_names:
            parts=re.split('__',var)
            if len(parts)>2:
                parts=['_'.join(parts[:-1]),parts[-1]]
            assert len(parts)==2,f'problem with parts of {var}'
            if not parts[0] in var_dict:
                var_dict[parts[0]]=[]
            var_dict[parts[0]].append((parts[1],var))
        return var_dict
        
        
    def missingVals(self):
        nan_01=self.X_nan_bool_df.to_numpy().astype(np.int8)
        feature_names=self.X_nan_bool_df.columns.to_list()
        feature_idx=np.arange(len(feature_names))
        #nan_bool_stack=self.X_nan_bool_df.reset_index(drop=True,inplace=False).to_numpy().astype(np.uint8)
        has_nan_features=nan_01.std(axis=0)>0
        nan_01_hasnan=nan_01[:,has_nan_features]
        hasnan_features=[name for i,name in enumerate(feature_names) if has_nan_features[i]] 
        nan_corr=self.pearsonCorrelationMatrix(nan_01_hasnan)
        nan_corr_df=pd.DataFrame(data=nan_corr, columns=hasnan_features)
        self.nan_corr=nan_corr
        self.nan_corr_df=nan_corr_df
        corr_linkage = hierarchy.ward(nan_corr)
        dendro = hierarchy.dendrogram( #just used for ordering the features by the grouping
            corr_linkage, labels=hasnan_features, ax=None,no_plot=True, leaf_rotation=90)
        feature_count=np.arange(nan_01.shape[1])
        
        plt.rcParams['font.size'] = '8'
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8),dpi=200)
        ax1.imshow(nan_01,aspect='auto',interpolation='none',cmap='plasma')
        colors=[plt.get_cmap('plasma')(value) for value in [255]]
        labels=['missing data']
        patches=[Patch(color=colors[i],label=labels[i]) for i in [0]]
        ax1.legend(handles=patches,bbox_to_anchor=(0.5,1.05),loc=9,ncol=2,fontsize='large')
        ax1.set_xticks(feature_idx)
        ax1.set_xticklabels(feature_names, rotation='vertical',fontsize=6)
        
        ax2.imshow(nan_corr[dendro['leaves'],:][:,dendro['leaves']],aspect='equal',interpolation='none')
        hasnan_feature_idx=np.arange(len(hasnan_features))
        ax2.set_yticks(hasnan_feature_idx)
        ax2.set_xticks(hasnan_feature_idx)
        ax2.set_xticklabels(dendro['ivl'], rotation='vertical',fontsize=6)
        ax2.set_yticklabels(dendro['ivl'],fontsize=6)
        ax2.set_title('Missing Data Clustering Across Features')
        fig.tight_layout()
    

    
    
    def hierarchicalDendrogram(self,linkage='ward',dist='spearmanr'):
        #from https://scikit-learn.org/dev/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py
        X=self.full_X_float_df#.to_numpy()
        #X=(X-X.mean())/X.std()
        plt.rcParams['font.size'] = '8'
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8),dpi=200)
        if dist.lower()=='spearmanr':
            corr = spearmanr(X,nan_policy='omit').correlation
        elif dist.lower()=='pearsonr':
            corr=self.pearsonCorrelationMatrix(X)
        else: assert False, 'distance not developed'
        if linkage.lower()=='ward':
            corr_linkage = hierarchy.ward(corr)
        else: assert False, 'linkage not developed'
        self.corr_linkage=corr_linkage
        self.corr=corr
        dendro = hierarchy.dendrogram(
            corr_linkage, labels=X.columns.tolist(), ax=ax1, leaf_rotation=90
        )
        dendro_idx = np.arange(0, len(dendro['ivl']))

        ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']],aspect='auto',interpolation='none')
        ax2.set_xticks(dendro_idx)
        ax2.set_yticks(dendro_idx)
        ax2.set_xticklabels(dendro['ivl'], rotation='vertical',fontsize=6)
        ax2.set_yticklabels(dendro['ivl'],fontsize=6)
        fig.tight_layout()
        plt.show()
        
    def pearsonCorrelationMatrix(self,Xdf):
        if type(Xdf) is pd.DataFrame:
            X=Xdf.to_numpy()
        else:
            X=Xdf
        cols=X.shape[1]
        corr_mat=np.empty((cols,cols))
        for c0 in range(cols):
            corr_mat[c0,c0]=1
            for c1 in range(cols):
                if c0<c1:
                    corr=pearsonr(X[:,c0],X[:,c1])[0]
                    if np.isnan(corr):
                        print(f'nan for {X[:,c0]} and {X[:,c1]}')
                    corr_mat[c0,c1]=corr
                    corr_mat[c1,c0]=corr
        return corr_mat