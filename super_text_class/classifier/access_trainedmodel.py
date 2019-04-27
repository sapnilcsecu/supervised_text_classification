'''
Created on Apr 23, 2019

@author: Nasir uddin
'''

import numpy as np  
import re  
from sklearn.feature_extraction.text import CountVectorizer
import pickle  
from nltk.corpus import stopwords  
import pickle 
def main():
#To load the model and predict
    with open('vocabulary_file', 'rb') as vocabulary_file:  
       count_vect= pickle.load(vocabulary_file)
    with open('text_classifier', 'rb') as training_model:  
        model = pickle.load(training_model)
    X= ["i have lost my debit card"]
    print(model.predict(count_vect.transform(X)))


if __name__ == '__main__':
     main()