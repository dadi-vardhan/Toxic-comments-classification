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

class Data_Preprocessing():
    def __init__(self):
        self.stemmer    =   PorterStemmer()
        self.lemmatizer =   WordNetLemmatizer()
        self.morphy_tag =   {'JJ':wordnet.ADJ,'VB':wordnet.VERB,'RB':wordnet.ADV}
        self.stop_words =   set(stopwords.words('english'))

    def remove_urls(self,comment:str):
        """remove urls present in a sentence.

        Args:
            comment (str): sentence to be cleaned

        Returns:
            str: [cleaned sentence.
        """
        return re.sub(r"http\S+", "", comment)
    
    def remove_alphanumeric(self,comment:str):
        """remove special/alphaneumeric characters from a sentence.

        Args:
            comment (str): sentence to be cleaned.

        Returns:
            str: [cleaned sentence.
        """
        return re.sub('[^A-Za-z0-9]+', ' ', comment)

    def convert_lowercase(self,comment:str):
        """convert words in a sentence to lowercase.

        Args:
            comment (str): sentence to be lowercased.

        Returns:
            str: lowercase sentence.
        """
        return str(comment).lower()
    
    def stemm_and_lemmatize(self,comment:str):
        """removes stopwords and perform stemming and lemmatization using pos tagging.

        Args:
            comment (str): sentence to be processed.

        Returns:
            str : processed sentence.
        """
        words   =   comment.split()
        processed_comment   =   []
        for word in words:
            if word not in self.stop_words:
                pos_tag =   nltk.pos_tag([str(word)])
                word    =   self.stemmer.stem(word)
                if pos_tag[0][1] in self.morphy_tag.keys():
                    word    =   self.lemmatizer.lemmatize(word,pos=self.morphy_tag[str(pos_tag[0][1])])
                    processed_comment.append(word)
                else:
                    word    =   self.lemmatizer.lemmatize(word)
                    processed_comment.append(word)

        processed_comment   =   ' '.join([w for w in processed_comment if len(w)>1])
        return  processed_comment
    
    def split_comment(self,comment:str):
        """Split sentences into words.

        Args:
            comment str: sentence to be splitted into to words.

        Returns:
            list[str]: list of words
        """
        return comment.split()
    
    def text_normalize(self,comment:str,tokens=True):
        """ Implement all the preprocessig steps.

        Args:
            comment (str): sentence to be preprocessed.
            tokens (bool, optional): If true returns tokens. Defaults to True.
        Returns:
            list: tokenized words. Default to True.
            str : preprocessed sentence.
        """
        comment =   self.remove_urls(comment)
        comment =   self.remove_alphanumeric(comment)
        comment =   self.convert_lowercase(comment)
        comment =   self.stemm_and_lemmatize(comment)
        words  =   self.split_comment(comment)
        if  tokens  ==  True:
            return words
        elif    tokens  ==  False:
            return  comment

if __name__=="__main__":

    TRAIN_PATH  =   'dataset_1/train.csv'
    TEST_PATH   =   'dataset_1/test.csv'

    # loading the data
    train_df    =   pd.read_csv(TRAIN_PATH, index_col='id')
    test_df     =   pd.read_csv(TEST_PATH,index_col='id')

    #preprocessign the data
    preprocessor    =   Data_Preprocessing()
    



