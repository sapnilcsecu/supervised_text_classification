'''
Created on Apr 23, 2019

@author: Nasir uddin
'''

import numpy as np  
import re  
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import pickle  
from nltk.corpus import stopwords  
import pickle 
def main():
#To load the model and predict
    with open('../vocabulary_file', 'rb') as vocabulary_file:  
       count_vect= pickle.load(vocabulary_file)
    with open('../text_classifier', 'rb') as training_model:  
        model = pickle.load(training_model)
    X= [" Great Read: I thought this book was brilliant, but yet realistic. It showed me that to error is human. I loved the fact that this writer showed the loving side of God and not the revengeful side of him. I loved how it twisted and turned and I could not put it down. I also loved The glass castle."]
    print(model.predict(count_vect.transform(X)))


if __name__ == '__main__':
     main()