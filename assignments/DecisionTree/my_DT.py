import pandas as pd
import numpy as np
from collections import Counter
from Node import Node

class my_DT:

    def __init__(self, criterion="entropy", max_depth=8, min_impurity_decrease=0, min_samples_split=2):
        #DID NOT USE HINT FILE
        # criterion = {"gini", "entropy"},
        # Stop training if depth = max_depth. Depth of a binary tree: the max number of edges from the root node to a leaf node
        # Only split node if impurity decrease >= min_impurity_decrease after the split
        #   Weighted impurity decrease: N_t / N * (impurity - N_t_R / N_t * right_impurity - N_t_L / N_t * left_impurity)
        # Only split node with >= min_samples_split samples
        self.criterion = criterion
        self.max_depth = int(max_depth)
        self.min_impurity_decrease = min_impurity_decrease
        self.min_samples_split = int(min_samples_split)
        self.features=None
        self.thresholds=None
        self.root=None


    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str
        self.classes_ = list(set(list(y)))
        # write your code below
        self.features=X.columns
        self.thresholds=self._getThresholds(X)
        y_counts=pd.Series(Counter(y))
        initial_P=1-sum((y_counts/sum(y_counts))**2)
        self.root=Node(X.copy(),y.copy(),depth=0,is_root=True,P=initial_P)
        self.build_tree(self.root)
        
    def build_tree(self, node):
        
        
        
        if node.depth>self.max_depth-1 :
            return
        
        else:
            flag=node.trySplit(features=self.features, criterion=self.criterion,split_thresholds=self.thresholds, min_samples_split=self.min_samples_split)
            
            if flag:
                if node.leftChild is not None:
                    self.build_tree(node.leftChild)
                if node.rightChild is not None:
                    self.build_tree(node.rightChild)
            else:
                node.leftChild=None
                node.rightChild=None
                return

    def print_tree(self, node):

        print(node)
        print(node.depth)
        print()

        if node.leftChild is not None:
                self.print_tree(node.leftChild)
        if node.rightChild is not None:
                self.print_tree(node.rightChild)
        
    def _getThresholds(self, X):
        
        thresholds={}
        for col in X.columns:
            thresholds[col]=X[col].copy().sort_values().unique()

        return thresholds
    
    
    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        # write your code below
        predictions=X.apply(self._predsPerRow, axis='columns')
        return predictions
    
    def _predsPerRow(self,row):
        node=self.root
        while True:
            if node.split_features is None:
                return node.getPred()
            if row[node.split_features[0]]>=node.split_features[1]:
                node=node.rightChild
            else:
                node=node.leftChild

    def _probsPerRow(self,row):
        
        node=self.root
        while True:
            if node.split_features is None:
                # if row['SepalLengthCm']==6.2:
                #     pdb.set_trace()
                return node.getProb()
            if row[node.split_features[0]]>=node.split_features[1]:
                node=node.rightChild
            else:
                node=node.leftChild
        
            

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # Eample:
        # self.classes_ = {"2", "1"}
        # the reached node for the test data point has {"1":2, "2":1}
        # then the prob for that data point is {"2": 1/3, "1": 2/3}
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        # write your code below

        ##################
        probs=X.apply(self._probsPerRow, axis='columns')
        probs.replace(np.NaN,0.0,inplace=True)
        return probs






        
        
        
    
    
