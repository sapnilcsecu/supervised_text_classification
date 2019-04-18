'''
Created on Apr 19, 2019

@author: Nasir uddin
'''


class test_input:
    '''
    classdocs
    '''

    def __init__(self, train_y, valid_y):
        self.train_y = train_y
        self.valid_y = valid_y
        
    def get_train_y(self):
        return self.train_y

    def get_valid_y(self):
        return self.valid_y
        
