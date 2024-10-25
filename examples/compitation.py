import pandas as pd
import pyarrow.parquet as pq
import os
from sklearn.tree import  DecisionTreeRegressor
from sklearn import preprocessing
import numpy as np
def read_train_data(path):
    appended_df = pd.DataFrame()
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".parquet"):  # Filter for text files (adjust as needed)
                file_path = os.path.join(root, file)
                df = pd.read_parquet(file_path, dtype_backend="pyarrow")
                df['date_id'] = df['date_id'].astype('int32')
                appended_df = pd.concat([appended_df, df])

    return appended_df;


train_df = read_train_data('/home/chandra/mldata/compitation/combination')
res_features = {}
print(train_df.columns)
if np.any(train_df.columns == 'partition_id'):
    train_df = train_df.drop(columns=['partition_id'])
for feature in train_df.columns :
    if feature.startswith('responder'):
        res_features[feature] = train_df[feature]
        train_df = train_df.drop(columns=[feature])

label_encoder = preprocessing.LabelEncoder()

train_df['symbol_id'] =label_encoder.fit_transform(train_df['symbol_id'])

test_df = read_train_data('/home/chandra/mldata/compitation/test.parquet/date_id=0')
test_df['symbol_id'] =label_encoder.fit_transform(test_df['symbol_id'])
if np.any(test_df.columns == 'is_scored'):
    test_df = test_df.drop(columns=['is_scored'])
if np.any(test_df.columns == 'row_id'):
    test_df = test_df.drop(columns=['row_id'])

current_array =[]

for key, value in res_features.items():
    regressor = DecisionTreeRegressor()
    regressor.fit(train_df, value)
    y_pred = regressor.predict(test_df)
    print(key)
    print(y_pred)
    if len(current_array) ==0:
        current_array =y_pred
    else:
        max_count =0
        y_pred_arry = y_pred.tolist()
        array = current_array.tolist()
        for index in range(0, len(array)-1):
            print(array[index])
            if y_pred_arry[index] >= array[index] :
                max_count = max_count +1

        if max_count >  (len(array)/2):
         current_array = y_pred

print(current_array)