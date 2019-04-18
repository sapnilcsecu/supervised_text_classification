'''
Created on Apr 18, 2019

@author: Nasir uddin
'''
from sklearn import model_selection,preprocessing
from model.prepare_data import prepare_data


class prepare_dataset:
    
    def prepare_data(self,trainDF):
        train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])
        # split the dataset into training and validation datasets 
        
        # label encode the target variable 
        encoder = preprocessing.LabelEncoder()
        train_y = encoder.fit_transform(train_y)
        valid_y = encoder.fit_transform(valid_y)
        # label encode the target variable 
        pra_object=prepare_data(train_x,valid_x,train_y,valid_y);
        return pra_object;