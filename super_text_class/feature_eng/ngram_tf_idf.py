'''
Created on Apr 19, 2019

@author: Nasir uddin
'''
from sklearn.feature_extraction.text import TfidfVectorizer
from model.Train_model_input import Train_model_input
from feature_eng.feature_eng import feature_eng
from model.test_input import test_input
from sklearn import model_selection

class ngram_tf_idf:
    '''
    classdocs
    '''


    def convert_feature(self,trainDF,train_x):
        train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])
        '''
        tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
        '''
        tfidf_vect_ngram = TfidfVectorizer(analyzer='word', ngram_range=(2,3), max_features=5000)
        tfidf_vect_ngram.fit(trainDF['text'])
        xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
        xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)
        test_input_ob=super().test_input_encode(train_y, valid_y)
        return Train_model_input( xtrain_tfidf_ngram, xvalid_tfidf_ngram,test_input_ob.get_train_y(), test_input_ob.get_valid_y())
        