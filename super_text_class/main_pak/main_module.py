'''
Created on Apr 17, 2019

@author: Nasir uddin
'''
from classifier.Naive_Bayes import Native_Bayes

def main():
   classifer=Native_Bayes();
   classifer.build_model() 
if __name__ == '__main__':
    main()