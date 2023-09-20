import numpy as np
import pandas as pd
from collections import Counter
import pdb
#USED HINT FILE
class my_evaluation:
    # Binary class or multi-class classification evaluation
    # Each data point can only belong to one class

    def __init__(self, predictions, actuals, pred_proba=None):
        # inputs:
        # predictions: list of predicted classes
        # actuals: list of ground truth
        # pred_proba: pd.DataFrame of prediction probability of belonging to each class
        self.predictions = np.array(predictions)
        self.actuals = np.array(actuals)
        self.pred_proba = pred_proba
        if type(self.pred_proba) == pd.DataFrame:
            self.classes_ = list(self.pred_proba.keys())
        else:
            self.classes_ = list(set(list(self.predictions) + list(self.actuals)))
        self.confusion_matrix = None
        
            
        
    def confusion(self):
        # compute confusion matrix for each class in self.classes_
        # self.confusion = {self.classes_[i]: {"TP":tp, "TN": tn, "FP": fp, "FN": fn}}
        # no return variables
        # write your own code below
        self.confusion_matrix={}
        for class_ in self.classes_:
            self.conf_populate(class_)        
            
        return

    def conf_populate(self,class_,preds=None):
            #Helper function to populate confusion matrix
            if preds is None:
                preds=self.predictions
            else:
                self.confusion_matrix={}
                
            
            positive_indices=np.argwhere(self.actuals==class_)
            positive_indices=positive_indices.reshape(len(positive_indices))
            predicted_positives= (preds==class_).sum()
            predicted_negatives=(preds!=class_).sum()
            self.confusion_matrix[class_]={}
            self.confusion_matrix[class_]['TP']=(self.actuals[positive_indices]==preds[positive_indices]).sum()
            self.confusion_matrix[class_]['FP']=predicted_positives-self.confusion_matrix[class_]['TP']
            self.confusion_matrix[class_]['FN']=len(positive_indices)-self.confusion_matrix[class_]['TP']
            self.confusion_matrix[class_]['TN']=predicted_negatives-self.confusion_matrix[class_]['FN']
            


    def precision(self, target=None, average = "macro"):
        # compute precision
        # target: target class (str). If not None, then return precision of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average precision
        # output: prec = float
        # note: be careful for divided by 0
        # write your own code below
        if self.confusion_matrix==None:
            self.confusion()
        try:
                if target is not None:
                    if target in self.classes_:
                        prec=self.confusion_matrix[target]['TP']/(self.confusion_matrix[target]['TP']+self.confusion_matrix[target]['FP'])
                    else:
                        print("Unknown Class")               
                else:
                    if(average=='macro'):
                        prec=sum([self.confusion_matrix[class_]['TP']/(self.confusion_matrix[class_]['TP']+self.confusion_matrix[class_]['FP']) 
                                for class_ in self.classes_])/3.0
                    if(average=='micro'):
                        total_TP=sum([self.confusion_matrix[class_]["TP"] for class_ in self.classes_])
                        total_FP=sum([self.confusion_matrix[class_]["FP"] for class_ in self.classes_])
                        prec=total_TP/(total_TP+total_FP)   
                    if(average=='weighted'):
                        prec=sum([sum(self.actuals==class_)*(self.confusion_matrix[class_]['TP'])/(self.confusion_matrix[class_]['TP']+self.confusion_matrix[class_]['FP']) 
                                for class_ in self.classes_])/len(self.actuals)
                return prec
            
        except ZeroDivisionError: print("Division by Zero, check/repopulate confusion matrix")


    def recall(self, target=None, average = "macro"):
        # compute recall
        # target: target class (str). If not None, then return recall of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average recall
        # output: recall = float
        # note: be careful for divided by 0
        # write your own code below
        if self.confusion_matrix==None:
            self.confusion()
        try:
            
                if target is not None:
                    if target in self.classes_:
                        rec=self.confusion_matrix[target]["TP"]/(self.confusion_matrix[target]["TP"]+self.confusion_matrix[target]["FN"])
                    else: 
                        print("Unknown Class")               
                else:
                    if(average=='macro'):
                        rec=sum([self.confusion_matrix[class_]["TP"]/(self.confusion_matrix[class_]["TP"]+self.confusion_matrix[class_]["FP"]) 
                                for class_ in self.classes_])/3.0
                    if(average=='micro'):
                        total_TP=sum([self.confusion_matrix[class_]["TP"] for class_ in self.classes_])
                        total_FN=sum([self.confusion_matrix[class_]["FN"] for class_ in self.classes_])
                        rec=total_TP/(total_TP+total_FN)   
                    if(average=='weighted'):
                        rec=sum([sum(self.actuals==class_)*(self.confusion_matrix[class_]["TP"])/(self.confusion_matrix[class_]["TP"]+self.confusion_matrix[class_]["FP"]) 
                                for class_ in self.classes_])/len(self.actuals)

                return rec
            
        except ZeroDivisionError: print("Division by Zero, check/repopulate confusion matrix")

    def f1(self, target=None, average = "macro"):
        # compute f1
        # target: target class (str). If not None, then return f1 of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average f1
        # output: f1 = float
        # note: be careful for divided by 0
        # write your own code below
        if self.confusion_matrix==None:
            self.confusion()
        try :
                if target is not None:
                    if target in self.classes_:
                        f1_score=2*self.confusion_matrix[target]["TP"]/(2*self.confusion_matrix[target]["TP"]+self.confusion_matrix[target]["FN"]+self.confusion_matrix[target]["FP"])
                    else:
                        print("Unknown Class")
                else:
                    if(average=='macro'):
                        f1_score=sum([2*(self.confusion_matrix[class_]["TP"])/(2*self.confusion_matrix[class_]["TP"]+self.confusion_matrix[class_]["FN"]+self.confusion_matrix[class_]["FP"]) 
                                for class_ in self.classes_])/3.0
                    if(average=='micro'):
                        total_TP=sum([self.confusion_matrix[class_]["TP"] for class_ in self.classes_])
                        total_FP=sum([self.confusion_matrix[class_]["FP"] for class_ in self.classes_])
                        total_FN=sum([self.confusion_matrix[class_]["FN"] for class_ in self.classes_])
                        f1_score=2*total_TP/(2*total_TP+total_FP+total_FN)   
                    if(average=='weighted'):
                        f1_score=sum([sum(self.actuals==class_) * 2 * self.confusion_matrix[class_]["TP"]/(2*self.confusion_matrix[class_]["TP"]+self.confusion_matrix[class_]["FN"]+self.confusion_matrix[class_]["FP"])
                                for class_ in self.classes_])/len(self.actuals)
                        
                return f1_score
            
        except ZeroDivisionError: print("Division by Zero, check/repopulate confusion matrix")


    def auc(self, target):
        # compute AUC of ROC curve for the target class
        # return auc = float
        # if target=='Iris-versicolor':
        #     pdb.set_trace()

        if type(self.pred_proba) == type(None):
            return None
        
        else:
            # write your own code below
            if type(self.pred_proba)==type(None):
                return None
            else:
                if target in self.classes_:
                    order = np.argsort(self.pred_proba[target])[::-1]
                    tp = 0
                    fp = 0
                    fn = Counter(self.actuals)[target]
                    tn = len(self.actuals) - fn
                    tpr = 0
                    fpr = 0
                    auc_target = 0
                    for i in order:
                        if self.actuals[i] == target:
                            tp += 1
                            fn -= 1
                            tpr = tp/(tp+fn) 
                        else:
                            fp += 1
                            tn -= 1
                            pre_fpr = fpr
                            fpr = fp/(fp+tn)
                            auc_target += tpr*(fpr-pre_fpr)
                else:
                    raise Exception("Unknown target class.")

                return auc_target
            
            
            
                

            
            # return auc_target

