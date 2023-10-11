from collections import Counter
import pandas as pd
import numpy as np
from metrics import giniIndex
import pdb

class Node(object):
    
    def __init__(self, X, y, depth, is_root=False, P=None, parent=None):
        
        self.split_features=None
        self.is_leaf=False
        self.is_root=is_root
        self.X=X
        self.y=y
        self.P=P
        self.parent=parent
        self.leftChild=None
        self.rightChild=None
        # self.counts_less=None
        # self.counts_more=None
        self.depth=depth

    def trySplit(self, features, split_thresholds=None,min_impurity_decrease=0, min_samples_split=2):
        
        if len(self.X)>=min_samples_split:
            
            best_scores={}
            #featureThreshList=[(feat,thresh) for feat in features for thresh in split_thresholds[feat]]
                    
           # pdb.set_trace()
            for feature in features:
                
                    
                    category_counts={(feature, thresh): [pd.Series(Counter(self.y[self.X[feature]>=thresh])),
                                                         pd.Series(Counter(self.y[self.X[feature]<thresh]))] 
                                                    for thresh in split_thresholds[feature] }
                    
                    scores=pd.Series({(f,t): giniIndex(category_counts[(f,t)]) for f,t in category_counts})
                    
                    
                    best_score_feat=min(scores)
                    
                    if best_score_feat<=min_impurity_decrease:
                        continue
                    else:
                        best_scores[scores.idxmin()]=best_score_feat
                        
            
            if best_scores:
                self.best_scores=best_scores
                best_scores=pd.Series(best_scores)
                P_value=self.P-min(best_scores)
                if P_value<0:
                    return False
                self.split_features=best_scores.idxmin()
                feat,thresh=self.split_features
                #self.counts_more,self.counts_less=category_counts[scores.idxmin()]
                leftX=self.X[self.X[feat]<thresh].copy()
                rightX=self.X[self.X[feat]>=thresh].copy()
                
                    
                if not leftX.empty:
                    self.leftChild=Node(X=leftX,y=self.y[self.X[feat]<thresh].copy(),parent=self,depth=self.depth+1, P=P_value)
                if not rightX.empty:
                    self.rightChild=Node(X=rightX,y=self.y[self.X[feat]>=thresh].copy(),parent=self,depth=self.depth+1, P=P_value)
                # if self.depth==1:
                #     pdb.set_trace()
                return True
            else:
                return False            
        
        else:
            self.is_leaf=True
            return False
            
    def getPred(self):

        return max(Counter(self.y))
    
    def getProb(self):
        counts=pd.Series(Counter(self.y))
        return counts/sum(counts)
        
    def __str__(self):

        return str(self.split_features) + "\n" + str(Counter(self.y)) 

    

           
                


