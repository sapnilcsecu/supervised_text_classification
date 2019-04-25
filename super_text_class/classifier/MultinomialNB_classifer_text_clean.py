'''
Created on Apr 24, 2019

@author: Nasir uddin
'''

##load the library
import pandas as pd
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import nltk.classify.util as util
import pickle 
##load the library

def main():
    
    documents = []
    #load the dataset
    df = pd.read_csv('Consumer_Complaints.csv')
    df.head()
    #load the dataset
    
     #add the input column and output column and clean up data
    col = ['Product', 'Consumer complaint narrative']
    df = df[col]
    df = df[pd.notnull(df['Consumer complaint narrative'])]
    df.columns = ['Product', 'Consumer_complaint_narrative']
    df['category_id'] = df['Product'].factorize()[0]
    category_id_df = df[['Product', 'category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'Product']].values)
    df.head()
    
    
    #add the input column and output column and clean up data

if __name__ == '__main__':
    main()