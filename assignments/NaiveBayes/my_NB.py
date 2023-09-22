import pandas as pd
import numpy as np
from collections import Counter

class my_NB:
    """Naive Bayes Classifier with alpha ranging from 0 to 1 which controls the smoothing factor"""
    def __init__(self, alpha=1):
        #DID NOT USE HINT FILE
        # alpha: smoothing factor
        # P(xi = t | y = c) = (N(t,c) + alpha) / (N(c) + n(i)*alpha)
        # where n(i) is the number of available categories (values) of feature i
        # Setting alpha = 1 is called Laplace smoothing
        self.alpha = alpha
        self.weights={}

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, str
        # y: list, np.array or pd.Series, dependent variables, int or str
        self.classes_ = list(set(list(y)))

        # Calculate P(yj) and P(xi|yj)        
        # make sure to use self.alpha in the __init__() function as the smoothing factor when calculating P(xi|yj)
        # write your code below
        if not isinstance(y,pd.Series):
            y=pd.Series(y,index=self.classes_)
        P_yj=y.value_counts()/len(y)
        self.weights['P_yj']=P_yj
        X_copy=X.copy()
        X_copy['y']=y
        y_groups=X_copy.groupby('y')
        for class_ in self.classes_:
            self.weights[class_]={}
            group=y_groups.get_group(class_)
            for xi in group:
                if xi=='y':continue
                p_xi=self._getProbXgivenY(xi=group[xi],unique_values=X[xi].unique())
                self.weights[class_][xi]=p_xi
        
        
    def _getProbXgivenY(self,xi,unique_values):
        

        vc=xi.value_counts()
        prob_xi=pd.Series(0,index=unique_values)
        prob_xi+=vc
        prob_xi.replace(np.NaN,0,inplace=True)
        prob_xi+=self.alpha
        prob_xi/=(len(xi)+self.alpha*len(unique_values))
        return prob_xi 


    def predict(self, X):
        # X: pd.DataFrame, independent variables, str
        # return predictions: list
        # write your code below
        
        probabilities=self.predict_proba(X)
        predictions=probabilities.idxmax(axis=1).tolist()
        return predictions
    
    def _probsPerRow(self,row):
        
        prob=[]
        for class_ in self.classes_:
            p=self.weights['P_yj'][class_]
            for k in row.keys():
                try:
                    p*=self.weights[class_][k][row[k]]
                except KeyError:
                   # pdb.set_trace()
                    p*=1
            
            prob.append(p)
        

        
        return np.array(prob)
    
        
    
    

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, str
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)                
        # P(yj|x) = P(x|yj)P(yj)/P(x)
        # write your code below
        
        prob=X.apply(self._probsPerRow,axis='columns')
        probs=pd.DataFrame(np.stack(prob),columns=self.classes_)
        sum=probs.sum(axis=1)
        probs=probs.div(sum,axis=0)
        
        return probs



