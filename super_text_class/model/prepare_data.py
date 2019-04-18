'''
Created on Apr 18, 2019

@author: Nasir uddin
'''

class prepare_data:
    '''
    classdocs
    '''


    def __init__(self, train_x, valid_x, train_y, valid_y):
        self.train_x=train_x
        self.valid_x=valid_x
        self.train_y=train_y
        self.valid_y=valid_y
        
    def gettrain_x(self):
         return self.train_x
       
    def gettrain_y(self):
        return self.train_y
    def getvalid_x(self):
        return self.valid_x
    def getvalid_y(self):
        return self.valid_y