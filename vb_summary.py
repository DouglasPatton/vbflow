from vb_helper import myLogger


class VBSummary(myLogger):
    def __init__(self):
        myLogger.__init__(self)
        
    def setData(self,full_x_float,full_y):
        self.X_float_df=full_x_float_df
        self.y_df=full_y_ser
        
    def summarize(self):
        self.hierarchicalDendogram()
    
    
    def hierarchicalDendogram(self):
        #from https://scikit-learn.org/dev/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py
        try: self.X_float_df
        except:self.floatifyX()
        X=self.X_float_df#.to_numpy()
        plt.rcParams['font.size'] = '8'
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8),dpi=200)
        corr = spearmanr(X,nan_policy='omit').correlation
        corr_linkage = hierarchy.ward(corr)
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
        
    