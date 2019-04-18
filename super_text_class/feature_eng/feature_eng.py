'''
Created on Apr 19, 2019

@author: Nasir uddin
'''
from sklearn import  preprocessing
from model.test_input import test_input


class feature_eng:
    '''
    classdocs
    '''

    def test_input_encode(self, train_y, valid_y):
        encoder = preprocessing.LabelEncoder()
        train_y = encoder.fit_transform(train_y)
        valid_y = encoder.fit_transform(valid_y)
        return test_input(train_y, valid_y)
