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

def merge_json_files(files):
    merged_data = pd.DataFrame()
    for file in files:
      with open(file, 'r') as f:
        data = pd.read_json(f)
        merged_data = pd.concat([merged_data, data], ignore_index=True)

    return merged_data


# Example usage
files_to_merge = ['/home/chandra/test_ka_1.json', '/home/chandra/test_ka_2.json', '/home/chandra/test_ka_3.json']
files_to_merge1 = ['/home/chandra/test_ka_4.json']
df = merge_json_files(files_to_merge)
df1 = merge_json_files(files_to_merge1)
df.describe()
df=df.drop(["Image_url",'Latitude', 'Village_Name','Longtitude', 'Survey_id', 'District_Name', 'Taluk_Name','Hobli_Name', 'Years', 'Season', 'Crop_Extent', 'CropSurveyDate', 'Weekname', 'Month'], axis=1)
df_crop = df1.loc[:, 'Crop_Extent']
df1=df1.drop(["Image_url",'Latitude', 'Village_Name','Longtitude', 'Survey_id', 'District_Name', 'Taluk_Name','Hobli_Name', 'Years', 'Season', 'Crop_Extent', 'CropSurveyDate', 'Weekname', 'Month'], axis=1)

label_encoder = preprocessing.LabelEncoder()
# crop_list = ["Fallow", "Maize-H","Betel Nuts (Areca nuts)",]
# for index,crop in enumerate(crop_list):
#data = pd.DataFrame({"District_code": "14","Taluk_code": "4","Hobli_code": "2","Village_code": "15","Year_code": "116","Season_code": "1", "Cropname": "Maize-L"}, index=[0]);
df1["District_code"] = label_encoder.fit_transform(df1["District_code"])
df1["Taluk_code"] = label_encoder.fit_transform(df1["Taluk_code"])
df1["Hobli_code"] = label_encoder.fit_transform(df1["Hobli_code"])
df1["Village_code"] = label_encoder.fit_transform(df1["Village_code"])
df1["Year_code"] = label_encoder.fit_transform(df1["Year_code"])
df1["Season_code"] = label_encoder.fit_transform(df1["Season_code"])
df1["Cropname"] = label_encoder.fit_transform(df1["Cropname"])



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



df1 = df1.replace('', np.nan)
df1.fillna(0, inplace=True)
df1['Cropname']  = df1['Cropname'].map(result)
y = df1.loc[:, 'Cropname']
x = df1.drop(['Cropname'], axis=1)
#print(x.values)
print(y.values)

#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
y_pred = model.predict(x)

print(y_pred)
area ={}
for index, value in enumerate(y_pred):
    if str(value) in area:
        area[str(value)].append(df_crop[index])
    else:
        area[str(value)] = []
        area[str(value)].append(df_crop[index])

print(area)