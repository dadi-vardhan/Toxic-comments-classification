import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Dense, Embedding, LSTM, Dropout
from sklearn.feature_extraction.text import TfidfVectorizer
import skmultilearn
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

class Logistic_Regression_Classifier():
    def __init__(self,train_data,test_data) -> None:
        self.TFIDF_Vectorizer   =   TfidfVectorizer(strip_accents='unicode', analyzer='word',\
                                                    ngram_range=(1,3),max_features=1000, \
                                                    norm='l2')
        
        self.train_data =   train_data
        self.test_data  =   test_data
        self.train_tfidf    =   None
        self.test_tfidf =   None
        self.LR_Classifier  =   BinaryRelevance(LogisticRegression(solver='liblinear'))
        self.pred_labels    =   None

    def vectorizer(self,data_train,data_test):
        """ perform tfidf vectorization and transform texts to one-hot-encoding.

        Args:
            data_train [strs]: training data
            data_test  [strs]: testing data
        """
        self.TFIDF_Vectorizer.fit_transform(data_train)
        self.train_tfidf    =   self.TFIDF_Vectorizer.transform(data_train)
        self.test_tfidf =   self.TFIDF_Vectorizer.transform(data_test)
    
    def fit_model(self,labels):
        """fit the model to the training data.

        Args:
            labels ([type]): [description]
        """
        self.LR_Classifier.fit(self.train_tfidf,labels)

    def predict_labels(self):
        """predict the labels.
        """
        self.y_labels   =   self.LR_Classifier.predict(self.test_tfidf)

    def evaluate_model(self,labels):
        """Finds accuracy, roc_auc_score, and logloss.

        Args:
            labels ([ints]): test labels

        Returns:
            accuracy: float : accuracy of the model.
            auc: float: 
        """
        self.pred_labels    =   self.LR_Classifier.predict(self.test_tfidf).todense()
        y_pred  =   self.LR_Classifier.predict_proba(self.test_tfidf).todense()
        accuracy    =   np.mean([accuracy_score(labels.iloc[:,i], \
                                                self.pred_labels[:,i]) for i in range (6)])
        auc =   np.mean([roc_auc_score(labels.iloc[:,i], y_pred[:,i]) for i in range (6)])
        logloss =   np.mean([log_loss(labels.iloc[:,i], y_pred[:,i]) for i in range(6)])
        return  accuracy,auc,logloss 

class LSTM_Classifier():
    def __init__(self,X_train_path,X_test_path,y_train_path,y_test_path) -> None:
        self.VOCAB_SIZE =   10000
        self.TOKENIZER  =   Tokenizer(num_words=self.VOCAB_SIZE)
        self.X_TRAIN_PATH   =   X_train_path
        self.X_TEST_PATH    =   X_test_path
        self.Y_TRAIN_PATH   =   y_train_path
        self.Y_TEST_PATH    =   y_test_path
        self.X_train    =   None
        self.Y_train    =   None
        self.X_test =   None
        self.Y_test =   None
        self.train_sequences    =   None
        self.test_sequences =   None
        self.PAD_LENGTH = 61
        self.train_padded   =   None
        self.test_padded    =   None
        self.model  =   Sequential()
        self.trained_model  =   None

    def load_data(self):
        """loads features and lables from pickle files.
        """
        self.X_train    =   pd.read_pickle(self.X_TRAIN_PATH)
        self.X_test =   pd.read_pickle(self.X_TEST_PATH)
        self.Y_train    =   pd.read_pickle(self.Y_TRAIN_PATH)
        self.Y_test =   pd.read_pickle(self.Y_TEST_PATH)
        
    def tokenize(self):
        """ tokenizes text and convert texts to seqences.
        """
        self.TOKENIZER.fit_on_texts(self.X_train["preprocessed_text"])
        self.train_sequences    =   self.TOKENIZER.texts_to_sequences(self.X_train["preprocessed_text"])
        self.test_sequences =   self.TOKENIZER.texts_to_sequences(self.X_test["preprocessed_text"])

    def pad_sequences(self):
        """ pads the sequnces.
        """
        self.train_sequences    =   pad_sequences(self.train_sequences,\
                                                    maxlen=self.PAD_LENGTH,\
                                                    truncating='post')
        self.test_sequences =   pad_sequences(self.test_sequences,\
                                                maxlen=self.PAD_LENGTH,\
                                                    truncating='post')

    def create_model(self):
        """ creates LSTM model.
        """
        self.model.add(Input(shape=(None,)))
        self.model.add(Embedding(input_dim=self.VOCAB_SIZE+1, output_dim=300,\
                                input_length=self.PAD_LENGTH, mask_zero=True))
        self.model.add(LSTM(units=50,dropout=0.4,return_sequences=False))
        self.model.add(Dense(512,activation='relu'))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(6, activation='sigmoid'))
        self.model.summary()

    def compile_and_fit(self):
        """ compiles and fits the model to the data.
        """
        self.model.compile(tf.keras.optimizers.Adam(lr = 5e-4),\
                            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),\
                            metrics=[tf.metrics.BinaryAccuracy(),\
                            tf.metrics.AUC(multi_label=True, name='auc')])
        
        callbacks    =   [tf.keras.callbacks.ModelCheckpoint('best_lstm_model', save_best_only=True),
                            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.2),
                            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)]

        lstm_history    =   self.model.fit(self.train_padded, self.y_train,\
                                    validation_data = (self.test_padded,self.y_test),\
                                    batch_size=32,epochs = 50,\
                                    callbacks = callbacks)

    def save_model(self):
        """ Saves the model after training.
        """
        self.model.save("lstm_toxic_comments_model")

    def load_model(self,model_name:str):
        self.trained_model  =   tf.keras.models.load_model(str(model_name))
    
    def evaluate_model(self):
        """Evaluates the current model with the tf's best lstm model.

        Returns:
            floats: returns loss accuracy and auc.
        """
        best_model  =   tf.keras.models.load_model('best_lstm_model')
        loss, accuracy, auc =   best_model.evaluate(self.test_padded,\
                                                        self.y_test)
        return loss, accuracy, auc

if __name__ ==  "__main__":

    #path to the pickle files with the preprocessed data
    X_TRAIN_PATH    =   "X_train.pkl"
    Y_TRAIN_PATH    =   "y_train.pkl"
    X_TEST_PATH =   "X_test.pkl"
    Y_TEST_PATH =   "y_test.pkl"


    ## LSTM classifier
    lstm_model = LSTM_Classifier(X_TRAIN_PATH,X_TEST_PATH,\
                                    Y_TRAIN_PATH,Y_TEST_PATH)

    #loading the preprocessed data
    lstm_model.load_data()

    #tokenize the sentnences and convert text to sequences
    lstm_model.tokenize()

    #pad the sequences
    lstm_model.pad_sequences()

    #create the lstm model
    lstm_model.create_model()

    #compile and fit the model

    lstm_model.compile_and_fit()

    #evaluate the model with tf best lstm model
    loss, accuracy, auc =   lstm_model.evaluate_model()

    print(f'Accuracy : {np.round(accuracy,4)}')
    print(f'Auc : {np.round(auc,4)}')
    print(f'Logloss : {np.round(loss,4)}')