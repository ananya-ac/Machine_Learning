import pandas as pd
import numpy as np
from collections import Counter



class my_KNN:
    #DID NOT USE HINT FILE
    def __init__(self, n_neighbors=5, metric="minkowski", p=2):
        # metric = {"minkowski", "euclidean", "manhattan", "cosine"}
        # p value only matters when metric = "minkowski"
        # notice that for "cosine", 1 is closest and -1 is furthest
        # therefore usually cosine_dist = 1- cosine(x,y)
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        if self.metric=='euclidean': p=2
        if self.metric=='manhattan': p=1
        self.p = p
        self.dataX=pd.DataFrame()
        self.dataY=pd.DataFrame()


         
    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str
        self.classes_ = list(set(list(y)))
        # write your code below
        self.dataX=X
        self.dataY=y


        return

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        # write your code below
        probabilities=self.predict_proba(X)
        predictions=probabilities.idxmax(axis=1).tolist()

        return predictions



    # def _minkowskiDistance(self,v,p):
        
    #     return (v**p)**(1/p)
    
    def _neighborcalcMink(self,row):
        #pdb.set_trace()
        
        neighbor_labels=self.dataY[(((np.abs(self.dataX-row))**self.p).sum(axis='columns')**(1/self.p)).sort_values().iloc[:self.n_neighbors].index]
        neighbor_S=self._nbrCalc(neighbor_labels)

        return neighbor_S

    def _nbrCalc(self,labels):
        neighbor_S=pd.Series({c:0 for c in self.classes_})
        neighbor_S+=pd.Series((Counter(labels)))
        neighbor_S/=self.n_neighbors
        neighbor_S.replace(np.nan,0,inplace=True)
        return neighbor_S
    
    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        # write your code below
        
        probs=pd.DataFrame()
        if self.metric == 'cosine':
            normTr=np.linalg.norm(self.dataX,ord=2,axis=1)
            normTr=np.expand_dims(normTr,1)
            normTe=np.linalg.norm(X,ord=2,axis=1)
            normTe=np.expand_dims(normTe,0)
            dotprod=self.dataX.dot(X.transpose())
            cosineSim=dotprod/(normTr@normTe)
            neighbours=[]
            for i in range(len(X)):
                neighbours.append(self._nbrCalc(self.dataY[cosineSim[i].sort_values(ascending=False).index[:self.n_neighbors]]))
            probs=pd.DataFrame(neighbours)

        if self.metric == 'minkowski' or self.metric=='euclidean' or self.metric=='manhattan':
            probs=X.apply(self._neighborcalcMink,axis='columns')
            

        return probs



