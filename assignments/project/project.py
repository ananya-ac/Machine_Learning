import pandas as pd
import time
import numpy as np 
import gensim.downloader
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import remove_stopwords, stem_text, strip_short, preprocess_string
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier, HistGradientBoostingClassifier
import pdb
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler


from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import RidgeClassifier
from sklearn.decomposition import SparsePCA, PCA, TruncatedSVD
 
class my_model():

    def __init__(self) -> None:
        
        self.clf=SGDClassifier()
        self.pipe=None
        #self.model=gensim.downloader.load('glove-twitter-100')
    
    def fit(self, X, y):
        # do not exceed 29 mins
        
        
        X_p=preprocessing(X)
        #self.clf.fit(X_p, y)
        ohe=OneHotEncoder()
        vect1=TfidfVectorizer()
        vect2=TfidfVectorizer()
        vect3=CountVectorizer()
        ct=make_column_transformer((ohe,['telecommuting','has_company_logo','has_questions']),
                                     (vect1, 'description'),
                                     (vect2, 'requirements'),
                                     (vect3, 'title'))
        
        ct=make_column_transformer((ohe,['telecommuting','has_company_logo','has_questions']),
                                    (vect1, 'description'),
                                    (vect2, 'requirements'))
    
        
        
        self.pipe=make_pipeline(ct,self.clf)
        
        #params={}
        # params['columntransformer__tfidfvectorizer-1__min_df']=[1,2,3,4,5]
        # params['columntransformer__tfidfvectorizer-1__lowercase']=[True, False]
        #params['columntransformer__tfidfvectorizer-1__stop_words']=['english',None]
        # params['columntransformer__tfidfvectorizer-2__min_df']=[1,2,3,4,5]
        # params['columntransformer__tfidfvectorizer-2__lowercase']=[True, False]
        #params['columntransformer__tfidfvectorizer-2__stop_words']=['english',None]
        # params['columntransformer__tfidfvectorizer-3__min_df']=[1,2,3,4,5]
        # params['columntransformer__tfidfvectorizer-3__lowercase']=[True, False]
        #params['columntransformer__tfidfvectorizer-3__stop_words']=['english',None]
        # params['sgdclassifier__loss']=['hinge','log_loss', 'perceptron']
        # params['sgdclassifier__penalty']=['l2','l1']
        # params['sgdclassifier__alpha']=uniform(0.0001,0.0005)
        
        # rand=RandomizedSearchCV(self.pipe, params, n_iter=10, cv=5, scoring='f1', random_state=42)
        # rand.fit(X_p,y)
        
        self.pipe.fit(X_p,y)

        

        return

    def predict(self, X):
        # remember to apply the same preprocessing in fit() on test data before making predictions
        
        X_p=preprocessing(X)
        return self.pipe.predict(X_p)


def preprocessing(X):
        
    X_copy=X.copy(deep=True)
    X_copy.drop('location', inplace=True, axis='columns')
    X_copy['description']=X_copy['description'].apply(clean)
    X_copy['title']=X_copy['title'].apply(stem_text)
    X_copy['requirements']=X_copy['requirements'].apply(clean)
    
    return X_copy 


def clean(string):
    #return strip_short(stem_text(string),minsize=5)
    return ' '.join(preprocess_string(string))


def preprocessing2(X, model):

    X_copy=X.copy(deep=True)
    X['description']=X_copy['description'].apply(clean)
    X['title']=X_copy['title'].apply(clean)
    X['requirements']=X_copy['requirements'].apply(clean)

    maxed_title=max([len(x) for x in X['title']])
    maxed_description=max([len(x) for x in X['description']])
    maxed_requirements=max([len(x) for x in X['requirements']])

    # title_model=Word2Vec(sentences=X_copy['title'], min_count=1)
    # req_model=Word2Vec(sentences=X_copy['requirements'], min_count=1)
    # desc_model=Word2Vec(sentences=X_copy['description'], min_count=1)

    # title_vectors=np.stack([vectorize(w, title_model, maxed_title) for w in X['title']])
    # desc_vectors=np.stack([vectorize(w, desc_model, maxed_description) for w in X['description']])
    # req_vectors=np.stack([vectorize(w, req_model, maxed_requirements) for w in X['requirements']])

    title_vectors=np.stack([pretrained_vectorize(w, maxed_title, model) for w in X['title']])
    desc_vectors=np.stack([pretrained_vectorize(w, maxed_description, model) for w in X['description']])
    req_vectors=np.stack([pretrained_vectorize(w, maxed_requirements, model) for w in X['requirements']])


    ohe=OneHotEncoder()
    
    ct=make_column_transformer((ohe, ['telecommuting','has_company_logo','has_questions']),
                                    remainder='drop')
    cat_fts=ct.fit_transform(X)

    features=np.concatenate([title_vectors, desc_vectors, req_vectors], axis=1)

    ss=StandardScaler()
    normalized_features=ss.fit_transform(features)
    final_features=np.concatenate([normalized_features, cat_fts], axis=1)
    
    return final_features

def pretrained_vectorize(sentence, pad_len, model):
    
    
    words_vecs = [model[word] for word in sentence if word in model ]
    
    if len(words_vecs) == 0:
        return np.zeros(100)
    if len(words_vecs) < pad_len:
        words_vecs + ((pad_len - len(words_vecs)) * [0])
    
    words_vecs = np.array(words_vecs)
    
    return words_vecs.mean(axis=0)

def vectorize(sentence, model, pad_len):
    
    words_vecs = [model.wv[word] for word in sentence if word in model.wv]
    
    if len(words_vecs) == 0:
        return np.zeros(100)
    
    if len(words_vecs) < pad_len:
        words_vecs + ((pad_len - len(words_vecs)) * [0])
    
    words_vecs = np.array(words_vecs)
    
    return words_vecs.avg(axis=0)
