from sys import modules
import torch
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
import sys
import numpy as np

sys.path.append('../')
from dendrites.preprocess_data import process
from dendrites.preprocess_data import map_labels
from dendrites.preprocess_data import drop_labels
from dendrites.preprocess_data import  replace_empty
from sklearn import  linear_model
from sklearn.impute import KNNImputer


input_data_frame = pd.read_csv("/home/chandra/mldata/house_pricing/train.csv")
quantity = pd.DataFrame(input_data_frame['SalePrice'])
input_data_frame = input_data_frame.drop(['SalePrice'], axis=1)
data_after_process = process("liner regression", input_data_frame,  ['Id'])
processed_data_frame = data_after_process['result']
print(processed_data_frame)
data_after_process1 = process("liner regression", quantity,  [])
quantity=data_after_process1['result']

predict_data_frame = pd.read_csv("/home/chandra/mldata/house_pricing/test.csv")
predict_data_frame = process("liner regression",predict_data_frame, ['Id'])

regression = linear_model.LinearRegression()
regression.fit(processed_data_frame, quantity['SalePrice'])
print(regression.predict(predict_data_frame['result']))