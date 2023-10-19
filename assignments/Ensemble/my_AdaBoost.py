import pandas as pd
import numpy as np
from copy import deepcopy

class my_AdaBoost:

    def __init__(self, base_estimator = None, n_estimators = 50):
        #DID NOT USE HINT FILE
        # Multi-class Adaboost algorithm (SAMME)
        # alpha = ln((1-error)/error)+ln(K-1), K is the number of classes.
        # base_estimator: the base classifier class, e.g. my_DT
        # n_estimators: # of base_estimator rounds
        self.base_estimator = base_estimator
        self.n_estimators = int(n_estimators)
        self.estimators = [deepcopy(self.base_estimator) for i in range(self.n_estimators)]
        self.alpha={}
        

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str

        self.classes_ = list(set(list(y)))
        k = len(self.classes_)
        error_limit=1-(1/k)
        weights=np.ones(len(X)) * 1/len(X)
        i=0
       
        while True:
            if i==self.n_estimators:
                break
            if i==0:
                 indices=np.array(range(len(X)))
            else:    
                indices=np.random.choice(range(135),size=135, p=weights, replace=True)
            sampledX=X.iloc[indices]
            sampledY=y.iloc[indices]
            self.estimators[i].fit(sampledX,sampledY)
            preds=self.estimators[i].predict(X)
            wrong_indices=np.unique(sampledX[~(preds==sampledY)].index)
            weighted_error_rate=sum(weights[wrong_indices])
            if weighted_error_rate>=error_limit:
                weights=np.ones(len(X)) * 1/len(X)
                continue
            else:
                self.alpha[i]=np.log((1-weighted_error_rate)/weighted_error_rate) + np.log(k-1)
                weights[wrong_indices]*=np.exp(self.alpha[i])
                weights/=sum(weights)
                i+=1

        return

        


    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        # write your code below
        
        probabilities=self.predict_proba(X)
        
        return probabilities.idxmax(axis=1).tolist()

    
   

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # prob: what percentage of the base estimators predict input as class C
        # prob(x)[C] = sum(alpha[j] * (base_model[j].predict(x) == C))
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        # write your code below

        preds={}
        class_dict=pd.get_dummies(self.classes_,dtype=int)
        
        for i,estimator in enumerate(self.estimators):
            preds[i]=(self.alpha[i]*class_dict[estimator.predict(X)].values).transpose()

        summation=np.zeros_like(preds[0])

        for pred in preds.values():
            summation+=pred
        
        summation=summation/(summation.sum(axis=1).reshape(len(summation),1))
        probs=pd.DataFrame(summation,columns=self.classes_)
        

        return probs





