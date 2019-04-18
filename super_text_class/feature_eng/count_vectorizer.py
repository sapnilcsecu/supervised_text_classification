'''
Created on Apr 19, 2019

@author: Nasir uddin
'''
from feature_eng.feature_eng import feature_eng
from sklearn.feature_extraction.text import  CountVectorizer
from model.Train_model_input import Train_model_input

class count_vectorizer(feature_eng):
    '''
    classdocs
    '''


    def convert_feature(self,trainDF,train_x,valid_x,train_y, valid_y):
         
        count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
        count_vect.fit(trainDF['text'])
        
        # transform the training and validation data using count vectorizer object
        xtrain_count =  count_vect.transform(train_x)
        xvalid_count =  count_vect.transform(valid_x)   
        test_input_ob=super().test_input_encode(train_y, valid_y)
        return Train_model_input( xtrain_count, xvalid_count,test_input_ob.get_train_y(), test_input_ob.get_valid_y())
    
        