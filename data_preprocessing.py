from posixpath import commonpath
import pandas as pd
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

train_df = pd.read_csv('dataset_1/train.csv', index_col='id')
test_df = pd.read_csv('dataset_1/test.csv',index_col='id')

class Data_Preprocessing():
    def __init__(self,train_path,test_path):
        self.stemmer=PorterStemmer()
        self.lemmatizer= WordNetLemmatizer()
        self.morphy_tag = {'NN':wordnet.NOUN,'JJ':wordnet.ADJ,'VB':wordnet.VERB,'RB':wordnet.ADV}

    def remove_urls(self,comment:str):
        return re.sub(r"http\S+", "", comment)
    
    def remove_alphanumeric(self,comment:str):
        return re.sub('[^A-Za-z0-9]+', ' ', comment)

    def convert_lowercase(self,comment:str):
        return str(comment).lower()
    
    def stem_and_lemmatize(self,comment:str):
        words = comment.split()
        processed_comment=[]
        for word in words:
            if word not in stopwords:
                pos_tag = nltk.pos_tag([str(word)])
                word = self.stemmer.stem(word)
                if pos_tag in self.morphy_tag.keys():
                    word = self.lemmatizer.lemmatize(word,pos=self.morphy_tag[str(pos_tag[0][1])])
                    processed_comment.append(word)
                else:
                    word = self.lemmatizer.lemmatize(word)
                    processed_comment.append(word)

        processed_comment = ' '.join(processed_comment)
        return  processed_comment
    
    def split_comment(slef,comment:str):
        return comment.split()
    
    def preprocess(self,comment:str):
        comment = self.remove_urls(comment)
        comment = self.remove_alphanumeric(comment)
        comment = self.convert_lowercase(comment)
        comment = self.stem_and_lemmatize(comment)
        tokens = self.split_comment(comment)
        return tokens






