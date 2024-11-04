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
from sklearn import  linear_model

class LinerRegression(nn.Module):
    def __init__(self, inputs, outputs):
        super(LinerRegression, self).__init__()
        self.liner =  nn.Linear(in_features=inputs.shape[1], out_features=1)
        self.inputs = inputs
        self.outputs = outputs

    def forward(self, x):
        return self.liner(x)

    def train(self, optimizer='SGD', learning_rate= 0.01, epoches=10):
        optimizerAlgo = torch.optim.SGD(model.parameters(), learning_rate)
        criterion = nn.MSELoss()
        if(optimizer == 'LBFGS'):
            optimizerAlgo = torch.optim.LBFGS(model.parameters())
        for epoche in range(epoches):
            inputs = torch.from_numpy(self.inputs.values).float()
            targets = torch.from_numpy(self.outputs.values).float()
            optimizerAlgo.zero_grad()
            outputs =  self.forward(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizerAlgo.step()

    def predict(self, x):
        predictable_inputs = Variable(torch.from_numpy(x.values.astype(np.float32)))
        result =  self.forward(predictable_inputs)
        print("result==>")
        print(result)
        return result

input_data_frame = pd.read_csv("C:\\Users\\parameswari\\Downloads\\Seasonwiseprocurementdetails2023_0.csv")
data_after_process = process("liner regression", input_data_frame, ['District', 'Commodity', 'Season'], ['No of Farmers', 'Amount(Rs)'])
processed_data_frame = data_after_process['result']
quantity = processed_data_frame['Qty(MTs)']
processed_data_frame = processed_data_frame.drop(['Qty(MTs)'], axis=1)
print(processed_data_frame.head())
model = LinerRegression(processed_data_frame, quantity)
print(model.liner.weight.shape)
model.train(epoches=1)
predict_data_frame =  pd.DataFrame({'District': ['CHITTOOR'],
'Commodity': ['Bajra'],  'Season':['Kharif-2021']})
predict_data_frame = map_labels(predict_data_frame, ['District', 'Commodity', 'Season'], data_after_process['map'])
print(model.predict(predict_data_frame).detach().numpy())
regression = linear_model.LinearRegression()
regression.fit(processed_data_frame, quantity)
print(regression.predict(predict_data_frame))