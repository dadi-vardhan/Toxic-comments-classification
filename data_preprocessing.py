#from posixpath import commonpath
import pandas as pd
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split

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
        MIN_WORD_LEN = 2
        words   =   comment.split()
        processed_comment   =   []
        for word in words:
            if word not in self.stop_words:
                pos_tag =   nltk.pos_tag([str(word)])
                word    =   self.stemmer.stem(word)
                if pos_tag[0][1] in self.morphy_tag.keys():
                    word    =   self.lemmatizer.lemmatize(word,\
                                    pos=self.morphy_tag[str(pos_tag[0][1])])
                    processed_comment.append(word)
                else:
                    word    =   self.lemmatizer.lemmatize(word)
                    processed_comment.append(word)

        processed_comment   =   ' '.join([w for w in processed_comment if len(w)>MIN_WORD_LEN])
        return  processed_comment
    
    def split_comment(self,comment:str):
        """Split sentences into words.

        Args:
            comment str: sentence to be splitted into to words.

        Returns:
            list[str]: list of words
        """
        return comment.split()

    def remove_white_spaces(slef,comment:str):
        pass 

    def remove_numbers(self,comment:str):
        processed_comment   =   ''.join(w for w in comment if not w.isdigit())
        return processed_comment

    def text_normalize(self,comment:str,tokens=False):
        """ Implement all the preprocessig steps.

        Args:
            comment (str): sentence to be preprocessed.
            tokens (bool, optional): If true returns tokens. Defaults to False.
        Returns:
            list: tokenized words. Default to False.
            str : preprocessed sentence.
        """
        comment =   self.remove_urls(comment)
        comment =   self.remove_alphanumeric(comment)
        comment =   self.convert_lowercase(comment)
        comment =   self.remove_numbers(comment)
        comment =   self.stemm_and_lemmatize(comment)
        words  =   self.split_comment(comment)
        if  tokens  ==  True:
            return words
        elif    tokens  ==  False:
            return  comment

if __name__=="__main__":

    TRAIN_PATH  =   'data/train.csv'
    TEST_PATH   =   'data/test.csv'

    # loading the data
    train_df    =   pd.read_csv(TRAIN_PATH, index_col='id')
    test_df     =   pd.read_csv(TEST_PATH,index_col='id')

    # rebalancing the data
    rowSums =   train_df.iloc[:,1:].sum(axis=1)
    rowSums =   rowSums.to_list()
    train_df["row_sums"]    =   rowSums
    grouped =   train_df.groupby(train_df.row_sums)
    df_non_toxic    =   grouped.get_group(0)
    mask    =   train_df["row_sums"]>0
    df_toxic    =   train_df[mask]
    df_non_toxic    =   df_non_toxic.sample(n=df_toxic.shape[0],random_state=42)
    new_train_df    =   df_toxic.append(df_non_toxic, ignore_index=True)
    new_train_df    =   new_train_df.sample(frac = 1)
    new_train_df    =   new_train_df.drop('row_sums', 1)

    #preprocessign the data
    preprocessor    =   Data_Preprocessing()
    train_df['preprocessed_text'] =   new_train_df['comment_text'].apply(preprocessor.text_normalize)
    test_df['preprocessed_text']  =   test_df['comment_text'].apply(preprocessor.text_normalize)

    #splitting training and testing data
    feature =   train_df[['preprocessed_text']]
    labels  =   train_df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]
    X_train, X_cv, y_train, y_cv    =   train_test_split(feature, labels)
    X_test  =   test_df[['preprocessed_text']]

    # storign the preprocessed data in pickle files (since data size is big)
    X_train.to_pickle('X_train.pkl')
    X_cv.to_pickle('X_cv.pkl')
    X_test.to_pickle('X_test.pkl')
    y_train.to_pickle('y_train.pkl')
    y_cv.to_pickle('y_cv.pkl')







