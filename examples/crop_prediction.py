import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
def get_model():
  model = keras.Sequential([
    Dense(9, input_shape=(17,), activation='relu'),
    Dense(1,  activation='relu')
  ])

  model.compile(
    loss='mse',
    optimizer='adam'
  )

  return model

def merge_json_files(files):
    merged_data = pd.DataFrame()
    for file in files:
      with open(file, 'r') as f:
        data = pd.read_json(f)
        merged_data = pd.concat([merged_data, data], ignore_index=True)

    return merged_data


# Example usage
files_to_merge = ['/home/chandra/test_ka_1.json', '/home/chandra/test_ka_2.json', '/home/chandra/test_ka_3.json',
                  '/home/chandra/test_ka_4.json']
df = merge_json_files(files_to_merge)
df.describe()
df=df.drop(["Image_url",'Latitude', 'Village_Name','Longtitude', 'Survey_id', 'District_Name', 'Taluk_Name','Hobli_Name', 'Years', 'Season', 'Crop_Extent', 'CropSurveyDate', 'Weekname', 'Month'], axis=1)
data_franes = pd.DataFrame([])
print(df.columns)
label_encoder = preprocessing.LabelEncoder()
# crop_list = ["Fallow", "Maize-H","Betel Nuts (Areca nuts)",]
# for index,crop in enumerate(crop_list):
data = pd.DataFrame({"District_code": "14","Taluk_code": "4","Hobli_code": "2","Village_code": "15","Year_code": "116","Season_code": "1", "Cropname": "Maize-L"}, index=[0]);
data["District_code"] = label_encoder.fit_transform(data["District_code"])
data["Taluk_code"] = label_encoder.fit_transform(data["Taluk_code"])
data["Hobli_code"] = label_encoder.fit_transform(data["Hobli_code"])
data["Village_code"] = label_encoder.fit_transform(data["Village_code"])
data["Year_code"] = label_encoder.fit_transform(data["Year_code"])
data["Season_code"] = label_encoder.fit_transform(data["Season_code"])
data["Cropname"] = label_encoder.fit_transform(data["Cropname"])



#print(data_franes)
df["District_code"] = label_encoder.fit_transform(df["District_code"])
df["Taluk_code"] = label_encoder.fit_transform(df["Taluk_code"])
df["Hobli_code"] = label_encoder.fit_transform(df["Hobli_code"])
df["Village_code"] = label_encoder.fit_transform(df["Village_code"])
df["Year_code"] = label_encoder.fit_transform(df["Year_code"])
df["Season_code"] = label_encoder.fit_transform(df["Season_code"])
#df["Cropname"] = label_encoder.fit_transform(df["Cropname"])

#print(df)
pd.set_option('future.no_silent_downcasting', True)
# Custom function to create a list of values within each group
def get_unique_values(group):
    dataMap ={}
    for index, value in enumerate(group):
         dataMap[value] = index;
    return dataMap;

# Apply the function to each group
result = get_unique_values(list(set(df['Cropname'])))
print(result)
df['Cropname']  = df['Cropname'].map(result)
df = df.replace('', np.nan)
df.fillna(0, inplace=True)
y = df.loc[:, 'Cropname']
x = df.drop(['Cropname'], axis=1)
#print(y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
#print(y_train.values)
model = OneVsRestClassifier(SVC())
model.fit(X_train, y_train)



data_franes = data.replace('', np.nan)
data_franes.fillna(0, inplace=True)
data['Cropname']  = data['Cropname'].map(result)
y = data.loc[:, 'Cropname']
x = data.drop(['Cropname'], axis=1)
#print(x.values)
print(y.values)

#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
y_pred = model.predict(x)

print(y_pred)
