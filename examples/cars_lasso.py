import  pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  Lasso
from sklearn import  preprocessing
import numpy as np


df = pd.read_csv("/home/chandra/mldata/lasso/auto-mpg.csv")
label_encoder = preprocessing.LabelEncoder()
df['model year']=label_encoder.fit_transform(df['model year'])
df['origin']=label_encoder.fit_transform(df['origin'])
df['car name'] = label_encoder.fit_transform(df['car name'])
df = df.replace('', np.nan)
df.fillna(0, inplace=True)
df = df.replace('?', np.nan)
df.fillna(0, inplace=True)
train, test = train_test_split(df, test_size=0.2)
train_y = train['mpg']
print(train.columns)
train_x = train.drop(columns=['mpg'])
lasso = Lasso()
lasso.fit(train_x, train_y)
test_y = test['mpg']
test_x = test.drop(columns=['mpg'])
print(lasso.predict(test_x))
print(test_y)