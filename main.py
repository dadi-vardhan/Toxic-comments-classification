import os
import sys
import pandas as pd


# loading the data
train_df = pd.read_csv('dataset_1/train.csv', index_col='id')
test_df = pd.read_csv('dataset_1/test.csv',index_col='id')