'''
Created on Apr 17, 2019

@author: Nasir uddin
'''
from sklearn import naive_bayes
from classifier.Naive_Bayes import Native_Bayes
from dataset_pre.dataset_load import dataset_load
from feature_eng.word_tf_idf import word_tf_idf
from classifier.Classifier import Classifier

def main():
    """
   classifer=Native_Bayes();
   classifer.build_model() 
   """
    load_data= dataset_load();
    
    trainDF=load_data.load_dataset()
    
    model_input=word_tf_idf().convert_feature(trainDF)
    
    # Naive Bayes on Word Level TF IDF Vectors
    accuracy = Classifier().train_model(naive_bayes.MultinomialNB(),model_input.get_train_input(), model_input.get_test_input(), model_input.get_train_target(),model_input.get_test_target())
    print ("NB, WordLevel TF-IDF: ", accuracy)
if __name__ == '__main__':
    main()