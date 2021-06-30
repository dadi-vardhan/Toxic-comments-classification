from classifiers import Logistic_Regression_Classifier, LSTM_Classifier
from data_preprocessing import Data_Preprocessing
import pickle
#path to the pickle files with the preprocessed data
X_TRAIN_PATH    =   "pkls/X_test.pkl"
Y_TRAIN_PATH    =   "pkls.y_train.pkl"
X_TEST_PATH =   "pkls/X_test.pkl"
Y_TEST_PATH =   "pkls/y_test.pkl"
LR_VECTORIZER_PATH   =   "pkls/LR_vectorizer.pkl"
LR_CLASSIFIER_PATH  =   "pkls/LR.pkl"

with open(LR_VECTORIZER_PATH, 'rb') as fin:
  vectorizer    =   pickle.load(fin)

with open(LR_CLASSIFIER_PATH, 'rb') as fin:
    clf =   pickle.load(fin)

if __name__  ==  '__main__':
    
    labels = [{'1':'toxic','0':0},{'1':'severe_toxic','0':0},
                {'1':'obscene','0':0},{'1':'threat','0':0},
                {'1':'insult','0':0},{'1':'identity_hate','0':0}]
    keep_in_loop    =   True
    while keep_in_loop  ==  True:
        print("Type 'exit' to end.")
        toxic_comment   =   str(input("Enter a comment >>> "))
        if toxic_comment    == 'exit':
            keep_in_loop = False
        else:
            pp = Data_Preprocessing()
            toxic_comment = pp.text_normalize(toxic_comment,tokens=False)
            tfidf_vec   =   vectorizer.transform([toxic_comment])
            predicted_labels    =   clf.predict(tfidf_vec).todense().tolist()
            predictions=[]
            for idx,val in enumerate(predicted_labels[0]):
                predictions.append(labels[idx][str(val)])
            predictions = [x for x in predictions if x !=0]
            if len(predictions)==0:
                print(" ")
                print("The comment is non-toxic")
                print(" ")
            else:
                print(" ")
                print("The comment is :",predictions)
                print(" ")
        



