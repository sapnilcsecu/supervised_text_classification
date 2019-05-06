'''
Created on May 6, 2019

@author: Nasir uddin
'''

from sklearn import naive_bayes
from dataset_pre.dataset_load import dataset_load
from feature_eng.word_tf_idf import word_tf_idf
from feature_eng.ngram_tf_idf import ngram_tf_idf
from feature_eng.count_vectorizer import count_vectorizer
from feature_eng.char_tf_idf import char_tf_idf
from classifier.Classifier import Classifier
from dataset_pre.prepare_dataset import  prepare_dataset


def main():
    
    # load the dataset
    load_data = dataset_load();
    trainDF = load_data.load_cvs_dataset("../corpus.csv")
    # load the dataset
    
    # Text Preprocessing
    txt_label = trainDF['label']
    txt_text = trainDF['text']
    clear_txt = prepare_dataset().clean_cvs_txt(txt_text)
    # Text Preprocessing
    
    # Text feature engineering with char_tf_idf 
    model_input = char_tf_idf().convert_feature(clear_txt, txt_label)
    # Text feature engineering 
    
    #  Build Text Classification Model and Evaluating the Model
    naive = naive_bayes.MultinomialNB()
    accuracy = Classifier().train_model(naive, model_input.get_train_input(), model_input.get_test_input(), model_input.get_train_target(), model_input.get_test_target())
    print ("NB, char_tf_idf accuracy is : ", accuracy * 100)
    
    # Text feature engineering with count_vectorizer
    model_input = count_vectorizer().convert_feature(clear_txt, txt_label)
    # Text feature engineering with count_vectorizer
    
    #  Build Text Classification Model and Evaluating the Model
    naive = naive_bayes.MultinomialNB()
    accuracy = Classifier().train_model(naive, model_input.get_train_input(), model_input.get_test_input(), model_input.get_train_target(), model_input.get_test_target())
    print ("NB, count_vectorizer accuracy is : ", accuracy * 100)
    #  Build Text Classification Model and Evaluating the Model
    
    # Text feature engineering with ngram_tf_idf
    model_input = ngram_tf_idf().convert_feature(clear_txt, txt_label)
    # Text feature engineering with ngram_tf_idf
    
    #  Build Text Classification Model and Evaluating the Model
    naive = naive_bayes.MultinomialNB()
    accuracy = Classifier().train_model(naive, model_input.get_train_input(), model_input.get_test_input(), model_input.get_train_target(), model_input.get_test_target())
    print ("NB, ngram_tf_idf accuracy is : ", accuracy * 100)
    #  Build Text Classification Model and Evaluating the Model
    
    # Text feature engineering with word_tf_idf
    model_input = word_tf_idf().convert_feature(clear_txt, txt_label)
    # Text feature engineering with word_tf_idf
    
    #  Build Text Classification Model and Evaluating the Model
    naive = naive_bayes.MultinomialNB()
    accuracy = Classifier().train_model(naive, model_input.get_train_input(), model_input.get_test_input(), model_input.get_train_target(), model_input.get_test_target())
    print ("NB, word_tf_idf accuracy is : ", accuracy * 100)
    #  Build Text Classification Model and Evaluating the Model
    
    
if __name__ == '__main__':
    main()
