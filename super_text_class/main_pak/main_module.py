'''
Created on Apr 17, 2019

@author: Nasir uddin
'''
from sklearn import naive_bayes
from dataset_pre.dataset_load import dataset_load
from feature_eng.word_tf_idf import word_tf_idf
from classifier.Classifier import Classifier
from dataset_pre.prepare_dataset import  prepare_dataset

def main():
    """
   classifer=Native_Bayes();
   classifer.build_model() 
   """
    load_data= dataset_load();
    
    trainDF=load_data.load_cvs_dataset_preprocess()
    txt_label=trainDF['label']
    txt_text=trainDF['text']
    clear_txt=prepare_dataset().clean_cvs_txt(txt_text)
    model_input=word_tf_idf().convert_feature(clear_txt,txt_label)
    
    # Naive Bayes on Word Level TF IDF Vectors
    accuracy = Classifier().train_model(naive_bayes.MultinomialNB(),model_input.get_train_input(),model_input.get_test_input(), model_input.get_train_target(), model_input.get_test_target())
    print ("NB, WordLevel TF-IDF: ", accuracy*100)
if __name__ == '__main__':
    main()