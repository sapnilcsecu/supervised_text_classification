'''
Created on Apr 19, 2019

@author: Nasir uddin
'''
from sklearn.feature_extraction.text import TfidfVectorizer
from feature_eng.feature_eng import feature_eng
from model.Train_model_input import Train_model_input
from sklearn import model_selection
class char_tf_idf(feature_eng):
    


    def convert_feature(self,trainDF):
        train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])
        tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
        tfidf_vect_ngram_chars.fit(trainDF['text'])
        xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
        xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x)
        test_input_ob=super().test_input_encode(train_y, valid_y)
        return Train_model_input( xtrain_tfidf_ngram_chars, xvalid_tfidf_ngram_chars,test_input_ob.get_train_y(), test_input_ob.get_valid_y())