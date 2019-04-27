'''
Created on Apr 10, 2019

@author: nasir_uddin
'''
##load the library
import pandas as pd
from io import StringIO
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import nltk.classify.util as util
from sklearn.metrics import accuracy_score
import pickle 
##load the library

def main():
    

    #load the dataset
    df = pd.read_csv('Consumer_Complaints.csv')
    df.head()
    #load the dataset
    #add the input column and output column and clean up data
    documents = []
    col = ['Product', 'Consumer complaint narrative']
    df = df[col]
    df = df[pd.notnull(df['Consumer complaint narrative'])]
    df.columns = ['Product', 'Consumer_complaint_narrative']
    df['category_id'] = df['Product'].factorize()[0]
    category_id_df = df[['Product', 'category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'Product']].values)
    stemmer = WordNetLemmatizer()
    df.head()
    #add the input column and output column and clean up data
    
    """
   # X_train, X_test, y_train, y_test = train_test_split(df['Consumer_complaint_narrative'], df['Product'], random_state = 0)
  
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    """
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(df['Consumer_complaint_narrative'])
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, df['Product'], random_state = 0)
    clf = MultinomialNB().fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(accuracy_score(y_test, y_pred)*100)  
    
    with open('vocabulary_file', 'wb') as vocabulary_file:  
        pickle.dump(count_vect,vocabulary_file)
    
    with open('text_classifier', 'wb') as picklefile:  
        pickle.dump(clf,picklefile)
    
    print(clf.predict(count_vect.transform(["This company refuses to provide me verification and validation of debt per my right under the FDCPA. I do not believe this debt is mine."])))
   # print(clf.predict(count_vect.transform(["i have lost my debit card"])))
    
    #print(clf.predict(count_vect.transform(["I am disputing the inaccurate information the Chex-Systems has on my credit report. I initially submitted a police report on XXXX/XXXX/16 and Chex Systems only deleted the items that I mentioned in the letter and not all the items that were actually listed on the police report. In other words they wanted me to say word for word to them what items were fraudulent. The total disregard of the police report and what accounts that it states that are fraudulent. If they just had paid a little closer attention to the police report I would not been in this position now and they would n't have to research once again. I would like the reported information to be removed : XXXX XXXX XXXX"])))
    
    #Saving and Loading the Model
   
#Saving and Loading the Model
   


if __name__ == '__main__':
     main()
