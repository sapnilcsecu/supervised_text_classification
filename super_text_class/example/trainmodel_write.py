'''
Created on May 7, 2019

@author: Nasir uddin
'''

from sklearn import naive_bayes
from dataset_pre.dataset_load import dataset_load
from feature_eng.word_tf_idf import word_tf_idf
from classifier.Classifier import Classifier
import pickle 

def main():
    
    #load the dataset
    load_data= dataset_load();
    trainDF=load_data.load_cvs_dataset("../corpus.csv")
    #load the dataset
    
    #Text Preprocessing
    
    txt_label=trainDF['label']
    txt_text=trainDF['text']
    
    #Text feature engineering 
    model_input=word_tf_idf().convert_feature(txt_text,txt_label)
    #Text feature engineering 
    
    #  Build Text Classification Model and Evaluating the Model
    naive=naive_bayes.MultinomialNB()
    accuracy = Classifier().train_model(naive,model_input.get_train_input(),model_input.get_test_input(), model_input.get_train_target(), model_input.get_test_target())
    print ("NB, word_tf_idf accuracy is : ", accuracy*100)
    
    
  
    with open('../vocabulary_file', 'wb') as vocabulary_file:  
        pickle.dump(model_input.gettfidf_vect(),vocabulary_file)
    
    with open('../text_classifier', 'wb') as picklefile:  
        pickle.dump(naive,picklefile)
    
    
    #  Build Text Classification Model and Evaluating the Model
if __name__ == '__main__':
    main()