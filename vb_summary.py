from vb_helper import myLogger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr,pearsonr
from scipy.cluster import hierarchy

class VBSummary(myLogger):
    def __init__(self):
        myLogger.__init__(self)
        
    def setData(self,df_dict):
        #data looks like: summary_data={'full_float_X':X_json_s,'full_y':y_json_s, 'X_nan_bool':X_nan_bool_s} 
        self.full_X_float_df=pd.read_json(df_dict['full_float_X'])
        self.full_y_df=pd.read_json(df_dict['full_y'],typ='series')
        self.X_nan_bool_df=pd.read_json(df_dict['X_nan_bool'])
        
        
    def missingVals():
        nan_01=self.X_nan_bool_df.to_numpy().astype(np.uint8)
        #nan_bool_stack=self.X_nan_bool_df.reset_index(drop=True,inplace=False).to_numpy().astype(np.uint8)
        plt.rcParams['font.size'] = '8'
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8),dpi=200)
        ax1.imshow(nan_01)
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
            assert False, 'pearsonr needs to be extended to handle matrix like spearmanr'
            corr=pearsonr(X).r
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

        ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
        ax2.set_xticks(dendro_idx)
        ax2.set_yticks(dendro_idx)
        ax2.set_xticklabels(dendro['ivl'], rotation='vertical',fontsize=6)
        ax2.set_yticklabels(dendro['ivl'],fontsize=6)
        fig.tight_layout()
        plt.show()