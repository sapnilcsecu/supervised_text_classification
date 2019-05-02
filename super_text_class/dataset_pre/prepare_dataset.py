'''
Created on Apr 18, 2019

@author: Nasir uddin
'''
from sklearn import model_selection,preprocessing
from model.prepare_data import prepare_data
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn


class prepare_dataset:
    """
    def prepare_data(self,txt_text,txt_label):
        train_x, valid_x, train_y, valid_y = model_selection.train_test_split(txt_text, txt_label)
        # split the dataset into training and validation datasets 
        
        # label encode the target variable 
        encoder = preprocessing.LabelEncoder()
        train_y = encoder.fit_transform(train_y)
        valid_y = encoder.fit_transform(valid_y)
        
        # Step - 2: Split the model into Train and Test Data set

        
        # label encode the target variable 
        pra_object=prepare_data(train_x,valid_x,train_y,valid_y)
        return pra_object
    """
   
    def clean_cvs_txt(self,txt_text):
        txt_text.dropna(inplace=True)

        # Step - 1b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
        txt_text = [entry.lower() for entry in txt_text]
        
        # Step - 1c : Tokenization : In this each entry in the corpus will be broken into set of words
        txt_text= [word_tokenize(entry) for entry in txt_text]
        
        # Step - 1d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
        
        # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
        tag_map = defaultdict(lambda : wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV
        documents=[]
        
        for index,entry in enumerate(txt_text):
            # Declaring Empty List to store the words that follow the rules for this step
            Final_words = []
            # Initializing WordNetLemmatizer()
            word_Lemmatized = WordNetLemmatizer()
            # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
            for word, tag in pos_tag(entry):
                # Below condition is to check for Stop words and consider only alphabets
                if word not in stopwords.words('english') and word.isalpha():
                    word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                    Final_words.append(word_Final)
            # The final processed set of words for each iteration will be stored in 'text_final'
            documents.append(str(Final_words))
            
        txt_text =documents    
        return txt_text