import pandas as pd
from collections import Counter
import numpy as np
   

def entropy(y):
    y=pd.Series(Counter(y))
    y/=y.sum()
    ylog=np.emath.logn(3,y)
    y*=ylog
    
    return -1*y.sum()



    return

def metricImprovement(node, featureThresh, metric):
    
    parent_metric=metric(node.y)
    feat,thresh=featureThresh
    left=node.y[node.X[feat]<thresh]
    right=node.y[node.X[feat]>=thresh]
    if left.empty:
        left_metric=0
    else: 
        left_metric=metric(left)
    if right.empty:
        right_metric=0
    else: 
        right_metric=metric(right)
    left_metric*=(len(left)/len(node.y))
    right_metric*=(len(right)/len(node.y))
    return parent_metric-(left_metric+right_metric)


def gini(y):
    y=pd.Series(Counter(y))
    y/=y.sum()
    return 1 - (y**2).sum()



    
