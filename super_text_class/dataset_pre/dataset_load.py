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

# load the dataset  
 
    def load_txt_dataset(self): 
        data = open("../corpus.txt",encoding="utf8").read()
        labels, texts = [], []
        for i, line in enumerate(data.split("\n")):
            content = line.split()
            labels.append(content[0])
            texts.append(" ".join(content[1:]))
        
        # create a dataframe using texts and lables
        trainDF = pandas.DataFrame()
        trainDF['text'] = texts
        trainDF['label'] = labels
        return trainDF
    # load the dataset
    
    
    def load_cvs_dataset(self,dataset_path):
        #Set Random seed
        np.random.seed(500)
        
        # Add the Data using pandas
        Corpus = pd.read_csv(dataset_path,encoding='latin-1')
        
        return Corpus