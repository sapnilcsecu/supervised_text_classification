'''
Created on Apr 18, 2019

@author: Nasir-uddin
'''



class Train_model_input:
    '''
    classdocs
    '''

    def __init__(self, train_input,test_input,train_target, test_target,tfidf_vect):
        self.train_input = train_input
        self.train_target = train_target
        self.test_input = test_input
        self.test_target = test_target
        self.tfidf_vect=tfidf_vect;
        
    def gettfidf_vect(self):
        return self.tfidf_vect
    def get_train_input(self):
        return self.train_input

    def get_train_target(self):
        return self.train_target

    def get_test_input(self):
        return self.test_input

    def get_test_target(self):
        return self.test_target    
