import pandas as pd
import numpy as np
import pdb

class my_KMeans:

    def __init__(self, n_clusters=8, init = "k-means++", n_init = 10, max_iter=300, tol=1e-4):
        #DID NOT USE HINT
        # init = {"k-means++", "random"}
        # use euclidean distance for inertia calculation.
        # stop when either # iteration is greater than max_iter or the delta of self.inertia_ is smaller than tol.
        # repeat n_init times and keep the best run (cluster_centers_, inertia_) with the lowest inertia_.

        self.n_clusters = int(n_clusters)
        self.init = init
        self.n_init = n_init
        self.max_iter = int(max_iter)
        self.tol = tol

        self.classes_ = range(n_clusters)
        # Centroids
        # Sum of squared distances of samples to their closest cluster center.
        self.inertia_ = None
        self.cluster_centers_={}

    def fit(self, X):
        # X: pd.DataFrame, independent variables, float        
        # repeat self.n_init times and keep the best run 
            # (self.cluster_centers_, self.inertia_) with the lowest self.inertia_.
        # write your code below
        cluster_centers={}
        for i in range(self.n_init):
            
            if self.init=='random':
                initial_centroids=X.iloc[np.random.choice(len(X),size=self.n_clusters,replace=False)].values
                
            if self.init=='k-means++':
                rand_index=np.random.choice(len(X))
                first_centroid=X.iloc[rand_index].values
                prob=self.euclidean_distance(X,first_centroid)/sum(self.euclidean_distance(X,first_centroid))
                other_centroids=X.iloc[np.random.choice(len(X),p=prob,size=self.n_clusters-1,replace=False)].values
                initial_centroids=np.vstack([first_centroid,other_centroids])

            centroids,inertia=self.get_clusters_(initial_centroids,X)
            cluster_centers[inertia]=centroids
        
        best_inertia=min(cluster_centers.keys())
        self.inertia_=best_inertia
        self.cluster_centers_=[row for row in cluster_centers[best_inertia]]
        return

    def euclidean_distance(self,a,b):
        return np.sum((np.array(a)-np.array(b))**2,axis=1)**0.5

    def get_clusters_(self,centroids,X):

        for i in range(self.max_iter):
                dists=[]
                
                for centroid in centroids:
                    
                    SE=self.euclidean_distance(X,centroid)
                    dists.append(SE)
                    
                dists_arr=np.array(dists).T
                X['cluster']=np.argmin(dists_arr,axis=1)
                inertia=0
                
                try:
                    for k in range(self.n_clusters):
                        inertia+=(self.euclidean_distance(X[X['cluster']==k].drop('cluster',axis='columns'),centroids[k])).sum()
                except:
                        inertia+=0                
                if i>0:
                    if prev_inertia-inertia<self.tol:
                        X.drop('cluster', inplace=True, axis='columns')
                        break
                
                prev_inertia=inertia
                centroids=np.array([X[X['cluster']==k].drop('cluster',axis='columns').mean() for k in range(len(centroids))])
                X.drop('cluster', inplace=True, axis='columns')
                
        return (centroids,inertia)

    

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        # write your code below
        dists=self.transform(X)
        predictions = [np.argmin(dist) for dist in dists]
        return predictions

    def transform(self, X):
        # Transform to cluster-distance space
        # X: pd.DataFrame, independent variables, float
        # return dists = list of [dist to centroid 1, dist to centroid 2, ...]
        # write your code below
        #pdb.set_trace()
        dists = [[np.linalg.norm(x-centroid) for centroid in self.cluster_centers_] for x in X.to_numpy()]
        return dists

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)





