from classifiers import Logistic_Regression_Classifier, LSTM_Classifier

#path to the pickle files with the preprocessed data
X_TRAIN_PATH    =   "X_train.pkl"
Y_TRAIN_PATH    =   "y_train.pkl"
X_TEST_PATH =   "X_test.pkl"
Y_TEST_PATH =   "y_test.pkl"

lstm_model = LSTM_Classifier(X_TRAIN_PATH,X_TEST_PATH,\
                                Y_TRAIN_PATH,Y_TEST_PATH)

lstm_model.load_model("lstm_toxic_cmnts")





