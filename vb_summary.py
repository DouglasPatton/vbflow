#NEEDS COMMENTS
from vb_helper import myLogger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import spearmanr,pearsonr
from scipy.cluster import hierarchy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
#from sklearn.pipeline import make_pipeline
import re #possibly used to search for double underscores

class VBSummary(myLogger):
    def __init__(self):
        myLogger.__init__(self)
        plt.rcParams['font.size'] = '8'
        
    def setData(self,df_dict):
        #data looks like: summary_data={'full_float_X':X_json_s,'full_y':y_json_s, 'X_nan_bool':X_nan_bool_s} 
        self.full_X_float_df=pd.read_json(df_dict['full_float_X'])
        self.full_y_df=pd.read_json(df_dict['full_y'])
        self.X_nan_bool_df=pd.read_json(df_dict['X_nan_bool'])
        
    def viewComponents(self,num_cols=[6,9],keep_cats=False):
        n=self.full_X_float_df.shape[0]
        k=self.full_X_float_df.shape[1]
        g=len(num_cols)
        fig=plt.figure(figsize=(4*g,12),dpi=200)
        cmap='cool'
        X=self.full_X_float_df
        for g_idx,col_count in enumerate(num_cols):
            ax=fig.add_subplot(g,1,g_idx+1,projection='3d')
            keep_cols=self.getTopNCols(col_count,keep_cats=keep_cats)
            X_scaled_expanded=StandardScaler().fit_transform(X.loc[(slice(None),keep_cols)])
            X_orthog=PCA(n_components=3).fit_transform(X_scaled_expanded)
            self.X_orthog=X_orthog
            
            sc=ax.scatter(*X_orthog.T,c=self.full_y_df.to_numpy(),cmap=cmap,s=4,marker='o',depthshade=False,alpha=0.5)
            if keep_cats:
                ax.set_title(f'PCA projection of top {col_count} columns')
            else:
                ax.set_title(f'PCA projection of top {col_count} numeric columns')
            clb=fig.colorbar(sc,shrink=0.25,orientation='horizontal')
            clb.ax.set_title('y')
            ax.set_xlabel('component 1')
            ax.set_ylabel('component 2')
            ax.set_zlabel('component 3')
        fig.tight_layout()
                
                
    def getTopNCols(self,n_cols,keep_cats=True):        
        
        try: self.spear_xy,self.r_list
        except:
            self.spear_xy=[];self.r_list=[]
        
            for col in self.full_X_float_df.columns:
                r=spearmanr(self.full_y_df,self.full_X_float_df[col]).correlation
                self.spear_xy.append((r,col))
                self.r_list.append(r)
        if keep_cats:
            r_arr=np.array(self.r_list)
        else:
            r_arr=np.array([r for r,col in self.spear_xy if not re.search('__',col)])
        #r_min=r_arr.mean()+r_arr.std()
        r_min=np.sort(np.abs(r_arr))[-n_cols]
        keep_cols=[]
        for r,col in self.spear_xy:
            if np.abs(r)>=r_min:
                keep_cols.append(col)
        return keep_cols
        
        
    def kernelDensityPie(self):
        try: self.spear_xy,self.r_list
        except: 
            _=self.getTopNCols(1)
        spear_xy_indexed=[
            (np.abs(tup[0]),tup[1],i) 
            for i,tup in enumerate(self.spear_xy)]
        abs_r_sort,col_sort,idx_sort=zip(
            *sorted(spear_xy_indexed,reverse=True))
        r_sort=[self.r_list[i] for i in idx_sort]  
            
        
        all_vars=col_sort
        float_vars,float_idx=zip(*[(name,i) for i,name in enumerate(all_vars) if not re.search('__',name)])
        if len(float_vars)<len(all_vars):
            cat_vars,cat_idx_list=zip(*[(name,i) for i,name in enumerate(all_vars) if not name in float_vars])
            
            cat_var_dict=self.mergeCatVars(cat_vars)
            cat_group_names=list(cat_var_dict.keys())
        else:
            cat_var_dict={}
        float_var_count=len(float_vars)
        total_var_count=float_var_count+len(cat_var_dict)+1 #dep var too
        
        plot_cols=int(total_var_count**0.5)
        plot_rows=-(-total_var_count//plot_cols) #ceiling divide
        fig,axes_tup=plt.subplots(nrows=plot_rows,ncols=plot_cols,figsize=(12,12),dpi=200)
        axes_list=[ax for axes in axes_tup for ax in axes]
        
        
        
        for ax_idx,ax in enumerate(axes_list):
                
            if ax_idx<float_var_count+1:
                if ax_idx==0:
                    self.full_y_df.plot.density(ax=ax,c='r',ind=200)
                else:
                    name=float_vars[ax_idx-1]
                    self.full_X_float_df.loc[:,[name]].plot.density(ax=ax,c='b',ind=200)
                    r=round(r_sort[float_idx[ax_idx-1]],2)
                    ax.set_title(f'rank correlation with y: {r}',fontsize='x-small')
                ax.legend(loc=1,bbox_to_anchor=(1,0.8),fontsize='x-small')
            elif ax_idx<total_var_count:
                cat_idx=ax_idx-float_var_count-1
                cat_name=cat_group_names[cat_idx]
                cat_flavors,var_names=zip(*cat_var_dict[cat_name])
                cum_r=np.sum(np.abs(np.array([r_sort[cat_idx_list[cat_vars.index(cat)]] for cat in var_names])))
                cat_df=self.full_X_float_df.loc[:,var_names]
                cat_df.columns=cat_flavors
                cat_shares=cat_df.sum()
                cat_shares.name=cat_name
                self.cat_shares=cat_shares
                cat_shares.plot(y=cat_name,ax=ax,kind='pie',fontsize='x-small')
                r=round(cum_r,2)
                ax.set_title(f'cumulative abs rank correlation with y: {r}',fontsize='x-small')
                #ax.legend(fontsize='x-small')
            else:
                ax.axis('off')
            #ax.text()
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
        n=self.X_nan_bool_df.shape[0]
        if np.sum(self.X_nan_bool_df.to_numpy().ravel())==0:
            print(f'no missing values found')
            return
        
        nan_01=self.X_nan_bool_df.to_numpy().astype(np.int16)
        feature_names=self.X_nan_bool_df.columns.to_list()
        feature_idx=np.arange(len(feature_names))
        #nan_bool_stack=self.X_nan_bool_df.reset_index(drop=True,inplace=False).to_numpy().astype(np.uint8)
        
        plt.rcParams['font.size'] = '8'
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(12, 16),dpi=200)
        feat_miss_count_ser=self.X_nan_bool_df.astype(np.int16).sum(axis=0)
        feat_miss_count_ser.plot.bar(ax=ax0,)
        ax0.set_title('Missing Data Counts by Feature')
        pct_missing_list=[f'{round(pct)}%' for pct in (100*feat_miss_count_ser/n).tolist()]
        self.addAnnotations(ax0,pct_missing_list)
        
        row_miss_count_ser=feat_miss_count_ser=self.X_nan_bool_df.astype(np.int16).sum(axis=1)
        ax1.bar(np.arange(n),row_miss_count_ser.to_numpy(),width=1)
        ax1.set_title('Missing Data Counts by Row')
        
        
        nan_01_sum=nan_01.sum(axis=0)
        has_nan_features=nan_01_sum>0
        nan_01_hasnan=nan_01[:,has_nan_features]
        hasnan_features=[name for i,name in enumerate(feature_names) if has_nan_features[i]] 
        nan_corr=self.pearsonCorrelationMatrix(nan_01_hasnan)
        nan_corr_df=pd.DataFrame(data=nan_corr, columns=hasnan_features)
        self.nan_corr=nan_corr
        self.nan_corr_df=nan_corr_df
        corr_linkage = hierarchy.ward(nan_corr)
        dendro = hierarchy.dendrogram( #just used for ordering the features by the grouping
            corr_linkage, labels=hasnan_features, ax=None,no_plot=True, leaf_rotation=90)
        
        ax2.imshow(nan_01,aspect='auto',interpolation='none',cmap='plasma')
        colors=[plt.get_cmap('plasma')(value) for value in [255]]
        labels=['missing data']
        patches=[Patch(color=colors[i],label=labels[i]) for i in [0]]
        ax2.legend(handles=patches,bbox_to_anchor=(0,1.1),loc=9,ncol=2,fontsize='large')
        ax2.set_xticks(feature_idx)
        ax2.set_xticklabels(feature_names, rotation='vertical',fontsize=6)
        ax2.set_title('Missing Data Layout')
        
        cp=ax3.imshow(nan_corr[dendro['leaves'],:][:,dendro['leaves']],aspect='equal',interpolation='none')
        fig.colorbar(cp,shrink=0.5)
        hasnan_feature_idx=np.arange(len(hasnan_features))
        ax3.set_yticks(hasnan_feature_idx)
        ax3.set_xticks(hasnan_feature_idx)
        ax3.set_xticklabels(dendro['ivl'], rotation='vertical',fontsize=6)
        ax3.set_yticklabels(dendro['ivl'],fontsize=6)
        ax3.set_title('Missing Data Clustering Across Features')
        fig.tight_layout()
    

    def addAnnotations(self,ax,notes):
        for i,p in enumerate(ax.patches):
            width = p.get_width()
            height = p.get_height()
            x, y = p.get_xy() 
            ax.annotate(notes[i], (x + width/2, y + height+1), ha='center',fontsize=6)
    
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

        cp=ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']],aspect='equal',interpolation='none')
        fig.colorbar(cp,shrink=0.5)
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