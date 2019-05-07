'''
Created on Apr 17, 2019

@author: Nasir uddin
'''
import pandas
import pandas as pd
import numpy as np


class dataset_load:
    '''
    classdocs
    '''


    
    
    def load_cvs_dataset(self,dataset_path,encoder):
        #Set Random seed
        np.random.seed(500)
        
        # Add the Data using pandas
        Corpus = pd.read_csv(dataset_path,encoding=encoder)
        #Corpus = pd.read_csv(dataset_path)
        
        return Corpus