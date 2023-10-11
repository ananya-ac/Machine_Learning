from collections import Counter
import pandas as pd
from metrics import *

class Node(object):
    
    def __init__(self, X, y, depth, is_root=False, P=None, parent=None):
        
        self.split_features=None
        self.is_root=is_root
        self.X=X
        self.y=y
        self.P=P
        self.parent=parent
        self.leftChild=None
        self.rightChild=None
        self.depth=depth

    def trySplit(self, features, criterion, split_thresholds=None,min_impurity_decrease=0, min_samples_split=2):
        
        if criterion=='gini':
             metric=gini
        if criterion=='entropy':
             metric=entropy
        if len(self.X)>=min_samples_split:
            
            
            featureThreshList=[(feat,thresh) for feat in features for thresh in split_thresholds[feat][1:-1]]
            feat,thresh=max(featureThreshList,key= lambda featureThresh:metricImprovement(self,featureThresh,metric))
            # if self.depth==2:
            #pdb.set_trace()
            metric_improvement=metricImprovement(self,(feat,thresh),metric)
            if metric_improvement<min_impurity_decrease:
                 return False
            else:
                self.split_features=(feat,thresh)
                leftX=self.X[self.X[feat]<thresh].copy()
                rightX=self.X[self.X[feat]>=thresh].copy()
                if not leftX.empty:
                        self.leftChild=Node(X=leftX,y=self.y[self.X[feat]<thresh].copy(),parent=self,depth=self.depth+1)
                if not rightX.empty:
                        self.rightChild=Node(X=rightX,y=self.y[self.X[feat]>=thresh].copy(),parent=self,depth=self.depth+1)
                return True
                    
        else:
            return False
           
           
    def getPred(self):

        return max(Counter(self.y))
    
    def getProb(self):
        counts=pd.Series(Counter(self.y))
        return counts/sum(counts)
        
    def __str__(self):

        return str(self.split_features) + "\n" + str(Counter(self.y)) 

    

           
                


