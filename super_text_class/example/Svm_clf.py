'''
Created on May 6, 2019

@author: Nasir uddin
'''

from sklearn import svm
from dataset_pre.dataset_load import dataset_load
from feature_eng.ngram_tf_idf import ngram_tf_idf
from classifier.Classifier import Classifier
from dataset_pre.prepare_dataset import  prepare_dataset

def main():
   
    #load the dataset
    load_data= dataset_load();
    trainDF=load_data.load_cvs_dataset("../corpus.csv")
    #load the dataset
    
    #Text Preprocessing
    txt_label=trainDF['label']
    txt_text=trainDF['text']
    clear_txt=prepare_dataset().clean_cvs_txt(txt_text)
    #Text Preprocessing
    
   
    #Text feature engineering 
    model_input=ngram_tf_idf().convert_feature(clear_txt,txt_label)
    #Text feature engineering 
    
    #  Build Text Classification Model and Evaluating the Model
    naive=svm.SVC()
    accuracy = Classifier().train_model(naive,model_input.get_train_input(),model_input.get_test_input(), model_input.get_train_target(), model_input.get_test_target())
    print ("NB, ngram_tf_idf accuracy is : ", accuracy*100)
   
    
if __name__ == '__main__':
    main()