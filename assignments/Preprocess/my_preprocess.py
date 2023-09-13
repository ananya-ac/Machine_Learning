import numpy as np
from scipy.linalg import svd


class my_normalizer:
    def __init__(self, norm="Min-Max", axis = 1):
        #     norm = {"L1", "L2", "Min-Max", "Standard_Score"}
        #     axis = 0: normalize rows
        #     axis = 1: normalize columns
        self.norm = norm
        self.axis = axis
        self.scalers={}

    def fit(self, X):
        #     X: input matrix
        #     Calculate offsets and scalers which are used in transform()
        X_array  = np.asarray(X)
        if self.axis==1:
            ax=0
        else:
            ax=1
        self.scalers.clear()   
        if self.norm=="Min-Max":
            x_max=X_array.max(axis=ax)
            x_min=X_array.min(axis=ax)
            self.scalers['x_max']=x_max
            self.scalers['x_min']=x_min
        if self.norm=="Standard_Score":
            mu=X_array.mean(axis=ax)
            sigma=X_array.std(axis=ax)
            self.scalers['mu']=mu
            self.scalers['sigma']=sigma
            
        if self.norm=="L1":
            l1=np.linalg.norm(X_array,ord=1,axis=ax)
            self.scalers['l1']=l1

        if self.norm=="L2":
            l2=np.linalg.norm(X_array,ord=2,axis=ax)
            self.scalers['l2']=l2
        
        # Write your own code below

    def transform(self, X, scalers=None):
        # Transform X into X_norm
        X_norm = np.asarray(X)
            
        if self.norm=="Min-Max":
            X_norm=(X_norm-self.scalers['x_min'])/(self.scalers['x_max']-self.scalers['x_min'])

        if self.norm=="Standard_Score":
            X_norm=(X_norm-self.scalers['mu'])/self.scalers['sigma']
        if self.norm=="L1":
            X_norm=X_norm/self.scalers['l1']
        if self.norm=="L2":
            X_norm=X_norm/self.scalers['l2']
            
        # Write your own code below
        return X_norm

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class my_pca:
    def __init__(self, n_components = 5):
        #     n_components: number of principal components to keep
        self.n_components = n_components
        self.principle_components=np.array([])
    def fit(self, X):
        #  Use svd to perform PCA on X
        #  Inputs:
        #     X: input matrix
        #  Calculates:
        #     self.principal_components: the top n_components principal_components
        U, s, Vh = svd(X)
        self.principle_components=Vh.transpose()
        # Write your own code below

    def transform(self, X):
        #     X_pca = X.dot(self.principal_components)
        X_array = np.asarray(X)
        X_pca=X_array.dot(self.principle_components)[:,:self.n_components] 
        # Write your own code below

        return X_pca

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def stratified_sampling(y, ratio, replace = True):
    #  Inputs:
    #     y: class labels
    #     0 < ratio < 1: number of samples = len(y) * ratio
    #     replace = True: sample with replacement
    #     replace = False: sample without replacement
    #  Output:
    #     sample: indices of stratified sampled points
    #             (ratio is the same across each class,
    #             samples for each class = int(np.ceil(ratio * # data in each class)) )

    if ratio<=0 or ratio>=1:
        raise Exception("ratio must be 0 < ratio < 1.")
    
    y_array = np.asarray(y)
    num_samples=np.ceil(len(y)*ratio)
    num_classes=len(np.unique(y_array))
    sample=[]
    samples_per_class={cl:np.argwhere(y_array==cl) 
                       for cl in np.unique(y_array)}
    
    for v in samples_per_class.values():
        sample.extend(np.random.choice(a=v.reshape(len(v)),size=int(np.ceil(num_samples/num_classes)),replace=replace))
        
    

    
    return np.random.permutation(np.array(sample)).astype(int)
