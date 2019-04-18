'''
Created on Apr 17, 2019

@author: Nasir uddin
'''
import pandas


class dataset_load:
    '''
    classdocs
    '''

# load the dataset  
 
    def load_dataset(self): 
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
   