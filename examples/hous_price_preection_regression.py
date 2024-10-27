import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.read_csv("/home/chandra/mldata/house_pricing/train.csv")
df = df.drop(columns=['Id'])
label_encoder = preprocessing.LabelEncoder()

def getMapata(list):
    data = {}
    for index,item in enumerate(list):
        data[item]=index

    return data;
def getMappedDf(df):
    string_columns = []
    for column in df.columns:
        if(df[column].dtype=='object'):
            string_columns.append(column)

    completeDataMap={}
    for index,column in enumerate(string_columns):
        completeDataMap[column]= getMapata(df[column])
        df[column] = df[column].map(completeDataMap[column])
getMappedDf(df)
df = df.replace('', np.nan)
df.fillna(0, inplace=True)
y= df.loc[:, 'SalePrice']
x= df.drop(columns=['SalePrice'])
model = LinearRegression();
model.fit(x.values, y.values)

df = pd.read_csv("/home/chandra/mldata/house_pricing/test.csv")
getMappedDf(df)
df = df.replace('', np.nan)
df.fillna(0, inplace=True)
idcol = df['Id']
df = df.drop(columns=['Id'])
y_preict = model.predict(df.values)
for index, id in enumerate(idcol):
    print("  {}  | {} ".format(id,y_preict[index]))
