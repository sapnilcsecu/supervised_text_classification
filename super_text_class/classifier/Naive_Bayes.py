'''
Created on Apr 16, 2019

@author: nasir-uddin
'''
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, numpy, textblob, string
#from keras.preprocessing import text, sequence
#from keras import layers, models, optimizers
from dataset_pre.dataset_load import dataset_load
from classifier.Classifier import Classifier 

class Native_Bayes(Classifier):

    def build_model(self):
        # split the dataset into training and validation datasets 
        load_data= dataset_load();
        
        trainDF=load_data.load_dataset()
        train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])
        # split the dataset into training and validation datasets 
        
        # label encode the target variable 
        encoder = preprocessing.LabelEncoder()
        train_y = encoder.fit_transform(train_y)
        valid_y = encoder.fit_transform(valid_y)
        # label encode the target variable 
        
        # word level tf-idf
        tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
        tfidf_vect.fit(trainDF['text'])
        xtrain_tfidf =  tfidf_vect.transform(train_x)
        xvalid_tfidf =  tfidf_vect.transform(valid_x)
        # word level tf-idf
        # ngram level tf-idf 
        tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
        tfidf_vect_ngram.fit(trainDF['text'])
        xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
        xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)
        # ngram level tf-idf 
        
        # characters level tf-idf
        tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
        tfidf_vect_ngram_chars.fit(trainDF['text'])
        xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
        xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x) 
        # characters level tf-idf
        
        # create a count vectorizer object 
        count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
        count_vect.fit(trainDF['text'])
        
        # transform the training and validation data using count vectorizer object
        xtrain_count =  count_vect.transform(train_x)
        xvalid_count =  count_vect.transform(valid_x)   
        
        # create a count vectorizer object 
        
        
        # Naive Bayes on Count Vectors
        accuracy = Classifier().train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count,valid_y)
        print ("NB, Count Vectors: ", accuracy)
        
        # Naive Bayes on Word Level TF IDF Vectors
        accuracy = Classifier().train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf,valid_y)
        print ("NB, WordLevel TF-IDF: ", accuracy)
        
        # Naive Bayes on Ngram Level TF IDF Vectors
        accuracy = Classifier().train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram,valid_y)
        print ("NB, N-Gram Vectors: ", accuracy)
        
        # Naive Bayes on Character Level TF IDF Vectors
        accuracy = Classifier().train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars,valid_y)
        print ("NB, CharLevel Vectors: ", accuracy)



