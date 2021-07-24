# Toxic-comments-classification

#Directory structure 

.
├── classifiers.py
├── data
│   ├── sample_submission.csv
│   ├── test.csv
│   ├── test_labels.csv
│   └── train.csv
├── data_preprocessing.py
├── file-structure.txt
├── LICENSE
├── lstm_toxic_cmnts
│   ├── assets
│   ├── keras_metadata.pb
│   ├── saved_model.pb
│   └── variables
│       ├── variables.data-00000-of-00001
│       └── variables.index
├── lstm_toxic_cmnts_os
│   ├── assets
│   ├── keras_metadata.pb
│   ├── saved_model.pb
│   └── variables
│       ├── variables.data-00000-of-00001
│       └── variables.index
├── lstm_toxic_cmnts_us
│   ├── assets
│   ├── keras_metadata.pb
│   ├── saved_model.pb
│   └── variables
│       ├── variables.data-00000-of-00001
│       └── variables.index
├── main.py
├── pkls
│   ├── LR.pkl
│   ├── LR_vectorizer.pkl
│   ├── LSTM_tokenizer.pkl
│   ├── X_test.pkl
│   ├── X_train.pkl
│   ├── y_test.pkl
│   └── y_train.pkl
├── pkls_os
│   ├── LR_os.pkl
│   ├── LR_os_vectorizer.pkl
│   ├── LSTM_os_tokenizer.pkl
│   ├── X_test_os.pkl
│   ├── X_train_os.pkl
│   ├── y_test_os.pkl
│   └── y_train_os.pkl
├── pkls_us
│   ├── LR_us.pkl
│   ├── LR_us_vectorizer.pkl
│   ├── LSTM_vectorizer_us.pkl
│   ├── X_test_us.pkl
│   ├── X_train_us.pkl
│   ├── y_test_us.pkl
│   └── y_train_us.pkl
├── __pycache__
│   ├── classifiers.cpython-38.pyc
│   └── data_preprocessing.cpython-38.pyc
├── README.md
└── testcases
    ├── __init__.py
    ├── __pycache__
    │   ├── __init__.cpython-38.pyc
    │   └── test_data_preprocessing.cpython-38-pytest-6.2.2.pyc
    └── test_data_preprocessing.py
    
# package dependency

1. keras - 2.5.0
2. Sklearn - 0.22.2.post1
3. numpu - 1.19.5
4. pandas - 1.1.5
5. nltk - 3.2.5
6. Regex - 2.2.1
7. scikit-multilearn-0.2.0

# To run
python main.py
